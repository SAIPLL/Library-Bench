import os, random
import time, json, argparse, threading
from typing import Dict, List
from src.utils import get_test_path, get_prediction_path, load_testset
from prompt import create_fewshot_prompt_nl2code
from verify import get_valid_solutions, wrap_check
from pathlib import Path
import concurrent.futures as cfuts
from tqdm import tqdm

import requests
import ollama
import anthropic
from openai import OpenAI
from google import genai

BATCH_SIZE = 10
MAX_RETRIES = 3

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
client_OA = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


odex_mappings = [json.loads(l) for l in open("odex_grammar_mapping.jsonl", "r")]


def get_grammar_file(library_lst):
    lib_str = ','.join(sorted(library_lst))
    for m in odex_mappings:
        if m["library"] == lib_str:
            return f'../../grammar/{m["grammar"]}'
    return ""


def get_grammar(file_name="question1-grammar.gbnf"):
    with open(file_name, 'r') as file:
        return file.read()


def get_response_claude(prompt, sample, verbose=False, grammar=False):
    if verbose: print(f"[prompt] \n{prompt}\n------")
    prompt = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = client.messages.create(
        model=args.model_name,
        messages=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    return [response.content[0].text]


def get_response_gpt(prompt, sample, verbose=False, grammar=False):
    if verbose: print(f"[prompt] \n{prompt}\n------")
    prompt = [{"role": "user", "content": prompt}]
    response = client_OA.chat.completions.create(
        model=args.model_name,
        messages=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"],
    )
    return [response.choices[i].message.content for i in range(args.n)]


def call_openai_api(system_prompt, prompt, temperature, model, stop) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            result = client_OA.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort="none"
            )
            break
        except Exception as e:
            print(e)
            time.sleep(10)

    finish_reason = result.choices[0].finish_reason
    if finish_reason not in ("stop", "length"):
        return f"ERROR:{finish_reason}"

    content = result.choices[0].message.content
    if not content:
        return "ERROR:empty_response"

    idx_code = content.find('<code>')
    if idx_code == -1:
        return content

    after_idx = idx_code + len('<code>')
    return content[after_idx:]

def call_gemini_api(system_prompt, prompt, temperature, model, stop) -> str:
    while True:
        try:
            interaction = client_gemini.interactions.create(
                model=model,  # e.g. "gemini-3.5-flash"
                system_instruction=system_prompt,
                input=prompt,
                store=False,  # <-- stateless: don't persist this interaction server-side
                generation_config={
                    "temperature": temperature,
                    #"max_output_tokens": max_tokens,
                    "stop_sequences": stop,
                    "thinking_level": "minimal",  # closest analog to reasoning_effort="none"; use "none" if your model/tier supports it
                }
            )
            break
        except Exception as e:
            import time; time.sleep(10)

    text = interaction.output_text
    if not text:
        return "ERROR:empty_response"

    idx_code = text.find('<code>')
    if idx_code == -1:
        return text

    after_idx = idx_code + len('<code>')
    return text[after_idx:]

def get_response_openai_api(prompt, sample, verbose=False, grammar=False):
    if verbose: print(f"[prompt] \n{prompt}\n------")
    s_prompt = 'Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things.'
    code = call_openai_api(
        system_prompt=s_prompt,
        prompt=prompt,
        temperature=args.temperature,
        model=args.model_name,
        stop=["</code>", "# SOLUTION END"],
    )
    return [code]

def get_response_gemini(prompt, sample, verbose=False, grammar=False):
    if verbose: print(f"[prompt] \n{prompt}\n------")
    s_prompt = 'Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things.'
    code = call_gemini_api(
        system_prompt=s_prompt,
        prompt=prompt,
        temperature=args.temperature,
        model=args.model_name,
        stop=["</code>", "# SOLUTION END"],
    )
    return [code]


def get_response_ollama_api(prompt, sample, verbose=False, grammar=False):
    if sample["suffix"] != "":
        prompt = prompt + sample["suffix"]
    try:
        response = ollama.generate(
            model=args.model_name,
            prompt=prompt,
            stream=False,
            keep_alive="6h",
            think=False,
            options={"num_ctx": 2048}
        )
        full_text = response['response']
        if '<code>' in full_text:
            return [full_text.split('<code>')[1].split('</code>')[0]]
        return [full_text]
    except Exception as e:
        return [f"Error: {str(e)}"]


def get_response_ollama(prompt, sample, verbose=False, grammar=False, timeout=60):
    if verbose: print(f"[prompt] \n{prompt}\n------")

    payload = {
        "model": args.model_name,
        "temperature": args.temperature,
        "prompt": ''.join(prompt),
        "stream": True,
        "options": {"num_ctx": 2048},
        "think": False,
        "keep_alive": -1
    }
    if args.top_p > 0:
        payload["options"]["top_p"] = args.top_p

    if grammar:
        grammar_file = get_grammar_file(sample['library'])
        if grammar_file:
            payload["grammar"] = get_grammar(grammar_file)

    result = []

    def do_request():
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=(10, timeout)
            )
            for line in response.iter_lines():
                if line:
                    try:
                        obj = json.loads(line)
                        result.append(obj.get("response", ""))
                    except json.JSONDecodeError:
                        pass
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            result.append("__timed_out__")

    t = threading.Thread(target=do_request)
    t.start()
    t.join(timeout=timeout)  # hard wall-clock limit

    if t.is_alive() or "__timed_out__" in result:
        return ["timed out"]

    return [''.join(result)]


RESPOND_DICT = {
    "gpt-4-0613": get_response_openai_api,
    "gpt-5.6-luna": get_response_openai_api,
    "gpt-3.5-turbo-0125": get_response_openai_api,
    "claude-3-opus-20240229": get_response_claude,
    "granite-code:latest": get_response_ollama,
    "gemma2:2b": get_response_ollama,
    "gemma2:9b": get_response_ollama,
    "qwen3:8b": get_response_ollama,
    "claude-haiku-4-5-20251001": get_response_claude,
    "claude-sonnet-4-6": get_response_claude,
    "gemini-3.8-flash": get_response_gemini,
    "gemini-3.5-flash-lite": get_response_gemini,
}

OLLAMA_MODELS = {k for k, v in RESPOND_DICT.items() if v is get_response_ollama}


def get_predictions(model_name, prompt, sample, index, sleep_time, verbose, grammar, timeout=60):
    if model_name in OLLAMA_MODELS:
        # pass timeout directly — no sleep_time loop needed, handled by retry
        return get_response_ollama(prompt, sample, verbose, grammar, timeout=timeout)

    if sleep_time == 0:
        return RESPOND_DICT[model_name](prompt, sample, verbose, grammar)

    predictions = None
    while predictions is None:
        try:
            predictions = RESPOND_DICT[model_name](prompt, sample, verbose, grammar)
            print(f"predicted at sample #{index}")
        except:
            print(f"sleep for {sleep_time} at sample #{index}")
            time.sleep(sleep_time)
    return predictions


def select_fewshot_examples(sample, candidates, num_examples=1, method="random"):
    if method == "random":
        num_examples = min(num_examples, len(candidates))
        return random.sample(candidates, num_examples)


def build_prompt(i, sample, dataset):
    """Build the prompt for a sample (pure, no side effects)."""
    examples = select_fewshot_examples(
        sample=sample,
        candidates=dataset[:i] + dataset[i+1:],
        num_examples=args.num_examples,
        method=args.fewshot_method,
    )
    prompt = create_fewshot_prompt_nl2code(
        sample=sample,
        examples=examples,
        num_tests=args.num_tests,
        function_name=args.function_name,
    )
    if args.strip_prompt:
        prompt = prompt.rstrip()
    return prompt


def process_sample(i, sample, dataset, timeout=60):
    """Process a single sample: build prompt, get predictions, return result."""
    prompt = build_prompt(i, sample, dataset)

    predictions = get_predictions(
        model_name=args.model_name,
        prompt=prompt, sample=sample, index=i,
        sleep_time=args.sleep_time, verbose=args.verbose,
        grammar=eval(args.grammar),
        timeout=timeout
    )

    print(f"predicted: {predictions[0]}\n")
    assert len(predictions) == args.n, "# num_valid"

    return {
        "predictions": predictions[0],
        "task_id": sample["task_id"],
        "library": sample["library"],
        "index": i   # carry index so we can retry with the right sample
    }


def run_batch(batch_items, dataset, timeout, output_path):
    """
    Submit a batch of (i, sample) tuples.
    Write successes to file, return timed-out items for retry.
    """
    timed_out = []

    with cfuts.ThreadPoolExecutor(max_workers=6) as executor:
        futs = {
            executor.submit(process_sample, i, sample, dataset, timeout): (i, sample)
            for i, sample in batch_items
        }

        for f in tqdm(cfuts.as_completed(futs), total=len(futs), desc=f"  timeout={timeout}s"):
            i, sample = futs[f]
            try:
                result = f.result()
            except Exception as e:
                print(f"  ✗ Exception on sample {i}: {e}")
                timed_out.append((i, sample))
                continue

            if result["predictions"] == "timed out":
                print(f"  ⚠ Sample {i} timed out.")
                timed_out.append((i, sample))
            else:
                result.pop("index", None)
                with open(output_path, 'a') as out:
                    out.write(json.dumps(result) + '\n')

    return timed_out


def get_completed_task_ids(output_path, dataset):
    from collections import Counter

    # Count how many times each task_id appears in the full dataset
    dataset_counts = Counter(
        sample["task_id"] for sample in dataset
        if "import " in sample["test_start"]
    )

    # Count how many times each task_id was successfully completed in the output
    completed_counts = Counter()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("predictions") != "timed out":
                        completed_counts[obj["task_id"]] += 1
                except json.JSONDecodeError:
                    pass

    # A task_id is fully done only if completed count >= dataset count
    return {tid for tid, count in dataset_counts.items() if completed_counts[tid] >= count}


def main():
    dataset = load_testset(args.input_path)
    args.verbose = False

    # ── Filter to samples with imports, skip already completed ────────────
    completed_ids = get_completed_task_ids(args.output_path, dataset)
    pending = [
        (i, sample) for i, sample in enumerate(dataset)
        if "import " not in sample["test_start"]
        and sample["task_id"] not in completed_ids
    ]

    #random.shuffle(pending)
    print(f"Total pending samples: {len(pending)}")

    is_ollama = args.model_name in OLLAMA_MODELS

    if not is_ollama:
        # Non-Ollama models: no batching or timeout logic needed
        with cfuts.ThreadPoolExecutor(max_workers=6) as executor:
            futs = {executor.submit(process_sample, i, s, dataset): (i, s) for i, s in pending}
            for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
                result = f.result()
                result.pop("index", None)
                with open(args.output_path, 'a') as out:
                    out.write(json.dumps(result) + '\n')
        return

    # ── Pass 1: sweep all batches with short timeout ───────────────────────
    print("\n═══ Pass 1: full sweep (fast timeout) ═══")
    all_timed_out = []

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        ids = [s["task_id"] for _, s in batch]
        print(f"\n─── Batch {batch_start // BATCH_SIZE + 1}: {ids} ───")
        timed_out = run_batch(batch, dataset, timeout=6000, output_path=args.output_path)
        all_timed_out.extend(timed_out)

    # ── Pass 2: retry timed-out samples with escalating timeouts ──────────
    if all_timed_out:
        print(f"\n═══ Pass 2: retrying {len(all_timed_out)} timed-out sample(s) ═══")
        retry_queue = all_timed_out

        for attempt in range(1, MAX_RETRIES + 1):
            timeout = 600 * (attempt + 1)   # 120s → 180s → 240s
            print(f"\n─── Retry {attempt}/{MAX_RETRIES} — timeout={timeout}s, "
                  f"{len(retry_queue)} sample(s) remaining ───")
            retry_queue = run_batch(retry_queue, dataset, timeout=timeout, output_path=args.output_path)
            if not retry_queue:
                break

        # Record permanent timeouts so they're skipped on future runs
        for i, sample in retry_queue:
            record = {"predictions": "timed out", "task_id": sample["task_id"], "library": sample["library"]}
            with open(args.output_path, 'a') as out:
                out.write(json.dumps(record) + '\n')
        if retry_queue:
            print(f"\n✗ Permanently timed out: {[s['task_id'] for _, s in retry_queue]}")

    print("\n✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=str, default="False")
    parser.add_argument("--language", type=str, default="en", choices=["en", "es", "ja", "ru"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="code-davinci-002",
        choices=["code-cushman-001", "code-davinci-001", "code-davinci-002",
                 "gpt-4-0613", "claude-3-opus-20240229", "gpt-3.5-turbo-0125",
                 "granite-code:latest", "gemma2:2b", "gemma2:9b", "qwen3:8b",
                 "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "gpt-5.6-luna", "gemini-3.8-flash", "gemini-3.5-flash-lite"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--sleep_time", type=int, default=30)
    parser.add_argument("--function_name", type=str, default="id", choices=["id", "constant", "intent"])
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--fewshot_method", type=str, default="random", choices=["random"])
    parser.add_argument("--strip_prompt", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    directory_path = f"data/nl2code/{args.model_name.replace(':','-')}"
    if Path(directory_path).is_dir():
        print(f"The directory '{directory_path}' exists.")
    else:
        os.makedirs(directory_path)
        print(f"The directory '{directory_path}' created.")

    if (not args.input_path) or (not args.output_path):
        if not args.language:
            raise Exception("Need to specify [language] or [i/o path]")
        if not args.input_path:
            args.input_path = get_test_path(args.language)
        if not args.output_path:
            args.output_path = get_prediction_path(
                args.model_name, args.language,
                args.num_examples, args.num_tests,
            )
            args.output_path = args.output_path.replace(':', '-')
            if eval(args.grammar):
                args.output_path = args.output_path.replace('.json', '-grammar.json')
            print(args.output_path)

    main()