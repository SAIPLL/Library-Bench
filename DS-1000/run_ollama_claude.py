import json
import os
import sys
import gzip
import argparse
import requests
import concurrent.futures as cfuts
import time
import re
import threading
import ollama
import random

from tqdm import tqdm

BATCH_SIZE = 10
MAX_RETRIES = 3  # max times to retry a batch that has timeouts


def call_ollama_api_api(system="", prompt="something", model="gemma2:2b", temperature=0.2):
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
        full_text = response['response']
        if '<code>' in full_text:
            return [full_text.split('<code>')[1].split('</code>')[0]]
        return [full_text]
    except Exception as e:
        return [f"Error: {str(e)}"]


def call_ollama_api(system="", prompt="something",
                    url="http://localhost:11434/api/generate",
                    grammar="", n_probs=0, top_p=0, temperature=0.2,
                    model="granite-code:latest", keep_alive=-1,
                    stop=[], max_tokens=1024, n=1, timeout=60, attempts=3):
    while True:
        time_dct = {
            "total_duration": 0, "load_duration": 0,
            "prompt_eval_count": 0, "prompt_eval_duration": 0,
            "eval_count": 0, "eval_duration": 0
        }
        try:
            payload = {
                "model": model,
                "temperature": temperature,
                "prompt": ''.join(prompt),
                "stream": True,
                "options": {},
                "system": system,
                "think": False,
                "keep_alive": keep_alive
            }
            payload["options"]["num_ctx"] = 2048
            if n_probs > 0:
                payload["options"]["n_probs"] = n_probs
            if top_p > 0:
                payload["options"]["top_p"] = top_p
            if len(grammar) > 0:
                payload["grammar"] = grammar

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=(10, timeout))
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return ["timed out"]

    json_objects = response.text.strip().split('\n')
    parsed_objects = [json.loads(obj) for obj in json_objects]
    r_array = []
    for i in parsed_objects:
        try:
            r_array.append(i["response"] if "response" in i else "")
        except:
            r_array.append("")

    try:
        time_dct = {
            "total_duration": parsed_objects[-1]["total_duration"],
            "load_duration": parsed_objects[-1]["load_duration"],
            "prompt_eval_count": parsed_objects[-1]["prompt_eval_count"],
            "prompt_eval_duration": parsed_objects[-1]["prompt_eval_duration"],
            "eval_count": parsed_objects[-1]["eval_count"],
            "eval_duration": parsed_objects[-1]["eval_duration"]
        }
    except:
        print("Error in time.")

    return [''.join(r_array)]


def get_completed_ids(model):
    """Return set of already-completed problem IDs and next file index."""
    problems_ids = set()
    last_idx = []
    folder_path = 'data/'
    for filename in os.listdir(folder_path):
        if f'{model}-grammar' in filename:
            file_path = os.path.join(folder_path, filename)
            match = re.search(r'\d+', filename)
            if match:
                last_idx.append(int(match.group()))
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    json_obj = json.loads(line.strip())
                    if json_obj['code'][0] != "timed out":
                        problems_ids.add(json_obj['metadata']['problem_id'])
    max_idx = str(max(last_idx) + 1) if len(last_idx) > 1 else "0"
    return problems_ids, max_idx


def get_grammar(library_name="Pandas"):
    file_name = f"../../grammar/files/{library_name.lower()}-atomimport-grammar.gbnf"
    with open(file_name, 'r') as file:
        return file.read()


def run_func(prompt, id, grammar, model, temperature, timeout=60, metadata=None):
    code = call_ollama_api(
        'Write a short code following the given format and indentation. Place the executable code between <code> '
        'and </code> tags, without any other non-executable things.',
        prompt,
        grammar=grammar,
        temperature=temperature,
        n=1,
        model=model,
        max_tokens=1024,
        stop=["</code>", "# SOLUTION END"],
        timeout=timeout
    )
    return {'id': id, 'code': code, 'metadata': metadata}


def run_batch(batch, model, temperature, with_grammar, target_library, file_output, timeout=60):
    """
    Submit a batch of problems, write successes to file.
    Returns list of problem dicts that timed out (need retry).
    """
    timed_out = []

    with cfuts.ThreadPoolExecutor(max_workers=6) as executor:
        futs = {}
        for p in batch:
            id = int(p['metadata']['problem_id'])
            prompt = p['prompt']
            library_needed = p['metadata']['library']

            if (library_needed.lower() == target_library) or target_library == "":
                grammar = get_grammar(library_needed) if with_grammar else ""
                fut = executor.submit(run_func, prompt, id, grammar, model, temperature, timeout=timeout, metadata=p['metadata'])
                futs[fut] = p

        for f in tqdm(cfuts.as_completed(futs), total=len(futs), desc="Batch progress"):
            result = f.result()
            if result['code'][0] == "timed out":
                print(f"  ⚠ Problem {result['id']} timed out — will retry.")
                timed_out.append(futs[f])
            else:
                with open(file_output, 'a') as out:
                    out.write(json.dumps(result) + '\n')

    return timed_out


def run_ollama(model, temperature, idx, with_grammar, target_library):
    # ── Load full dataset ──────────────────────────────────────────────────
    all_problems = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]
    total = len(all_problems)
    print(f"Total problems in dataset: {total}")

    # ── Skip already-completed problems ───────────────────────────────────
    completed_ids = set()
    if with_grammar:
        completed_ids, idx = get_completed_ids(model)
        print(f"Already completed: {len(completed_ids)} problems")

    pending = [
        p for p in all_problems
        if int(p['metadata']['problem_id']) not in completed_ids
        and ((p['metadata']['library'].lower() == target_library) or target_library == "")
    ]
    random.shuffle(pending)
    print(f"Problems remaining: {len(pending)}")

    # ── Output file ────────────────────────────────────────────────────────
    suffix = "grammar" if with_grammar else "no-grammar"
    file_output = f'data/{model}-{suffix}-{idx}-answers.jsonl'
    print(f"Output file: {file_output}")

    # ── Pass 1: run all batches with short timeout, collect timeouts ──────
    print("\n═══ Pass 1: full sweep (fast timeout) ═══")
    all_timed_out = []

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        batch_ids = [int(p['metadata']['problem_id']) for p in batch]
        print(f"\n─── Batch {batch_start // BATCH_SIZE + 1}: problems {batch_ids} ───")
        timed_out = run_batch(batch, model, temperature, with_grammar, target_library, file_output, timeout=60)
        all_timed_out.extend(timed_out)

    # ── Pass 2: retry all timeouts with progressively longer timeouts ─────
    if all_timed_out:
        print(f"\n═══ Pass 2: retrying {len(all_timed_out)} timed-out problem(s) ═══")
        retry_queue = all_timed_out
        for attempt in range(1, MAX_RETRIES + 1):
            timeout = 60 * (attempt + 1)   # 120s, 180s, 240s …
            print(f"\n─── Retry {attempt}/{MAX_RETRIES} — timeout={timeout}s, "
                  f"{len(retry_queue)} problem(s) remaining ───")
            retry_queue = run_batch(retry_queue, model, temperature, with_grammar, target_library, file_output, timeout=timeout)
            if not retry_queue:
                break

        if retry_queue:
            print(f"\n✗ Permanently timed out after {MAX_RETRIES} retries: "
                  f"{[int(p['metadata']['problem_id']) for p in retry_queue]}")
            for p in retry_queue:
                record = {'id': int(p['metadata']['problem_id']),
                          'code': ["timed out"],
                          'metadata': p['metadata']}
                with open(file_output, 'a') as out:
                    out.write(json.dumps(record) + '\n')

    print("\n✓ All batches complete.")


if __name__ == "__main__":
    """
    Example:
        python run_ollama.py --grammar "False" --library ""
        python run_ollama.py --grammar "True" --model gemma2:2b
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="granite-code:latest")
    parser.add_argument("--idx",     type=str, default="0",    help="output file index")
    parser.add_argument("--grammar", type=str, default="False")
    parser.add_argument("--library", type=str, default="",     help="filter to a specific DS1000 library")
    args = parser.parse_args()

    run_ollama(args.model, temperature=1, idx=args.idx,
               with_grammar=eval(args.grammar), target_library=args.library)