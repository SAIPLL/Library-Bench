import json
import os
import sys
import argparse
import requests
import concurrent.futures as cfuts
import time
import re
import threading
import random
import pandas as pd
import ollama
from anthropic import Anthropic

from tqdm import tqdm

BATCH_SIZE = 10
MAX_RETRIES = 3

client_anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def get_dataset_hf():
    splits = {'v0.1.0_hf': 'data/v0.1.0_hf-00000-of-00001.parquet', 'v0.1.1': 'data/v0.1.1-00000-of-00001.parquet', 'v0.1.2': 'data/v0.1.2-00000-of-00001.parquet', 'v0.1.3': 'data/v0.1.3-00000-of-00001.parquet', 'v0.1.4': 'data/v0.1.4-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/bigcode/bigcodebench/" + splits["v0.1.0_hf"])
    return df


def call_anthropic_api(system_prompt, prompt, temperature, n, model, max_tokens) -> list[str]:
    prompt = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    while True:
        try:
            result = client_anthropic.messages.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system_prompt
            )
            break
        except:
            time.sleep(10)

    stop_reason = result.stop_reason
    if stop_reason not in ("end_turn", "stop_sequence", "max_tokens"):
        return [f"ERROR:{stop_reason}"]
    if not result.content:
        return ["ERROR:empty_response"]

    code = result.content[0].text
    idx_code = code.find('<code>')
    code = code[idx_code + len('<code>'):]
    return [code]


def call_ollama_api(system="", prompt="something",
                    url="http://localhost:11434/api/generate",
                    grammar="", n_probs=0, top_p=0, temperature=0.2,
                    model="granite-code:latest", keep_alive=-1,
                    stop=[], max_tokens=1024, n=1, timeout=60):
    time_start = time.time()

    payload = {
        "model": model,
        "temperature": temperature,
        "prompt": ''.join(prompt),
        "stream": True,
        "options": {"num_ctx": 2048},
        "think": False,
        "keep_alive": keep_alive,
    }
    if n_probs > 0:
        payload["options"]["n_probs"] = n_probs
    if top_p > 0:
        payload["options"]["top_p"] = top_p
    if len(grammar) > 0:
        payload["grammar"] = grammar
    if len(system) > 0:
        payload["system"] = system

    result = []

    def do_request():
        try:
            response = requests.post(
                url, data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                stream=True, timeout=(10, timeout)
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

    duration = time.time() - time_start

    if t.is_alive() or "__timed_out__" in result:
        return ["timed out"], duration

    return [''.join(result)], duration


def call_ollama_api_api(system="", prompt="something", model="gemma2:2b", temperature=1.0):
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            system=system,
            stream=False,
            keep_alive="6h",
            think=False,
            options={"num_ctx": 2048}
        )
        return [response['response']]
    except Exception as e:
        return [f"Error: {str(e)}"]


def get_completed_ids_file(file_name):
    completed_files = []
    for ix, l in enumerate(open(file_name, "r")):
        try:
            completed_files.append(json.loads(l))
        except:
            print(f"JSON decode error at line {ix}")
    return {c['task_id'] for c in completed_files if c['solution'][0] != 'timed out'}


def find_grammar_bigcodebench(bigcodebench_mappings, library_name):
    lib_str = library_name.replace('[', '').replace(']', '').replace(' ', '').replace("'", '')
    lib_str = ','.join(sorted(lib_str.split(',')))
    if lib_str in bigcodebench_mappings:
        id = bigcodebench_mappings[lib_str]['id']
        file_grammar = f'../../grammar/files/bigcodebench/{id}-atomimport-grammar.gbnf'
        with open(file_grammar, 'r') as f:
            return f.read()
    return ""


def get_grammar(library_name="Pandas"):
    file_name = f"../../grammar/files/{library_name.lower()}-reorder-grammar.gbnf"
    with open(file_name, 'r') as f:
        return f.read()


def run_func_ollama(prompt, task_id, model, temperature, grammar_st="", timeout=60):
    """Ollama HTTP path — returns timed out sentinel on timeout."""
    code, duration = call_ollama_api(
        '',
        prompt,
        grammar=grammar_st,
        temperature=temperature,
        model=model,
        max_tokens=1024,
        timeout=timeout
    )
    return {'task_id': task_id, 'solution': code, 'duration': duration}


def run_func_api(prompt, task_id, model, temperature):
    """Claude / ollama-library path — no timeout handling needed."""
    if 'claude' in model:
        code = call_anthropic_api(
            'Place the executable code between tags, without any other non-executable things',
            prompt,
            temperature=temperature,
            n=1,
            model=model,
            max_tokens=4096
        )
    else:
        code = call_ollama_api_api('', prompt, model=model, temperature=temperature)
    return {'task_id': task_id, 'solution': code}


def run_batch(batch_rows, model, temperature, grammar, bigcodebench_mappings, file_output, timeout=60):
    """
    Submit a batch of DataFrame rows.
    Write successes to file, return timed-out rows for retry.
    """
    timed_out = []

    with cfuts.ThreadPoolExecutor(max_workers=5) as executor:
        futs = {}
        for _, task_info in batch_rows:
            task_id = task_info['task_id']
            prompt = task_info['instruct_prompt']
            grammar_st = find_grammar_bigcodebench(bigcodebench_mappings, task_info['libs']) if eval(grammar) else ""
            fut = executor.submit(run_func_ollama, prompt, task_id, model, temperature, grammar_st, timeout)
            futs[fut] = (_, task_info)

        for f in tqdm(cfuts.as_completed(futs), total=len(futs), desc=f"  timeout={timeout}s"):
            _, task_info = futs[f]
            try:
                result = f.result()
            except Exception as e:
                print(f"  ✗ Exception on {task_info['task_id']}: {e}")
                timed_out.append((_, task_info))
                continue

            if result['solution'][0] == "timed out":
                print(f"  ⚠ {result['task_id']} timed out.")
                timed_out.append((_, task_info))
            else:
                with open(file_output, 'a') as out:
                    out.write(json.dumps(result) + '\n')

    return timed_out


def run_ollama(model, temperature, idx, completed_set, bigcodebench_mappings, grammar, file_output):
    # ── Build pending list ─────────────────────────────────────────────────
    pending = [
        (i, task_info)
        for i, task_info in bigcodebench.iterrows()
        if 0 <= i < 1200 and task_info['task_id'] not in completed_set
    ]
    random.shuffle(pending)
    print(f"Pending samples: {len(pending)}")

    is_ollama_http = 'claude' not in model  # adjust if you add more API models

    if not is_ollama_http:
        # API models: no batching/timeout logic
        with cfuts.ThreadPoolExecutor(max_workers=5) as executor:
            futs = {
                executor.submit(run_func_api, task_info['instruct_prompt'], task_info['task_id'], model, temperature): task_info
                for _, task_info in pending
            }
            for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
                result = f.result()
                with open(file_output, 'a') as out:
                    out.write(json.dumps(result) + '\n')
        return

    # ── Pass 1: sweep all batches with short timeout ───────────────────────
    print("\n═══ Pass 1: full sweep (fast timeout) ═══")
    all_timed_out = []

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        ids = [t['task_id'] for _, t in batch]
        print(f"\n─── Batch {batch_start // BATCH_SIZE + 1}: {ids} ───")
        timed_out = run_batch(batch, model, temperature, grammar, bigcodebench_mappings, file_output, timeout=60)
        all_timed_out.extend(timed_out)

    # ── Pass 2: retry timed-out samples with escalating timeouts ──────────
    if all_timed_out:
        print(f"\n═══ Pass 2: retrying {len(all_timed_out)} timed-out sample(s) ═══")
        retry_queue = all_timed_out

        for attempt in range(1, MAX_RETRIES + 1):
            timeout = 60 * (attempt + 1)   # 120s → 180s → 240s
            print(f"\n─── Retry {attempt}/{MAX_RETRIES} — timeout={timeout}s, "
                  f"{len(retry_queue)} sample(s) remaining ───")
            retry_queue = run_batch(retry_queue, model, temperature, grammar, bigcodebench_mappings, file_output, timeout=timeout)
            if not retry_queue:
                break

        for _, task_info in retry_queue:
            record = {'task_id': task_info['task_id'], 'solution': ['timed out'], 'duration': -1}
            with open(file_output, 'a') as out:
                out.write(json.dumps(record) + '\n')
        if retry_queue:
            print(f"\n✗ Permanently timed out: {[t['task_id'] for _, t in retry_queue]}")

    print("\n✓ Done.")


if __name__ == "__main__":
    """ python run_ollama_claude.py --model="qwen3:8b" --grammar="True" """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="granite-code:latest")
    parser.add_argument("--idx",     type=str, default="0")
    parser.add_argument("--grammar", type=str, default="False")
    args = parser.parse_args()

    bigcodebench = get_dataset_hf()

    bigcodebench_mappings = {}  # populate as needed

    model    = args.model
    temp     = 1
    grammar  = args.grammar

    file_output = f'results/{model.replace(":", "-")}-{args.idx}-constrained-answers.jsonl'
    os.makedirs('results', exist_ok=True)
    print(f"Output: {file_output}")

    completed_set = get_completed_ids_file(file_output) if os.path.exists(file_output) else set()
    print(f"Already completed: {len(completed_set)}")

    run_ollama(model, temp, args.idx, completed_set, bigcodebench_mappings, grammar, file_output)