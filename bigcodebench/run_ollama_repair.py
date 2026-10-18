import json
import os
import sys
import gzip
import argparse
import requests
import concurrent.futures as cfuts
import time
import re

from tqdm import tqdm


def get_completed_ids_file(file_name):
    completed_files = []
    ix = 0
    for l in open(file_name, "r"):
        try:
            c = json.loads(l) 
        except:
            print(ix)
        completed_files.append(c)
        ix = ix +1   
    completed_set = set()
    for c in completed_files:
        completed_set.add((c['task_id'],c['model']))
    return completed_set

def call_ollama_api(system="", prompt="something", 
                    url="http://localhost:11434/api/generate", 
                    grammar="", n_probs=0, top_p=0, temperature=0.2, 
                    model="granite-code:latest", keep_alive=-1,
                    stop=[], max_tokens=1024, n=1, timeout=60, attempts=3):
    """
        Call the model with either stop token,
        or n look-ahead tokens
    """
    time_start = time.time()
    while True:
        try:
            # Get call
            payload = {
                "model": model,
                "temperature": temperature,
                "prompt": ''.join(prompt),
                "stream": True,
                "options": {},
                "keep_alive": keep_alive
            }
            
            if n_probs > 0:
                payload["options"]["n_probs"] = n_probs
            if top_p > 0:
                payload["options"]["top_p"] = top_p
            if len(grammar) > 0:
                payload["grammar"] = grammar
            if len(system) > 0:
                payload["system"] = system
            
            headers = {
                "Content-Type": "application/json"
            }
            #x = x + 1
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
            break
        except requests.exceptions.Timeout:
            time.sleep(10);
            #if x < attempts:
                #continue
            #else:
            return ["timed out"]
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # e.g., 404 Client Error: Not Found
            print(f"Response content: {http_err.response.text}")  # body of the response
        except requests.exceptions.RequestException as err:
            print(f"Other error occurred: {err}")  # network issues, connection errors, etc.
    
    time_end = time.time()
    duration = time_end - time_start
    # Split the string into individual JSON objects
    json_objects = response.text.strip().split('\n')
    # Parse each JSON object
    parsed_objects = [json.loads(obj) for obj in json_objects]
    r_array = []
    for i in parsed_objects:
        try:
            if "response" in i:
                r_array.append(i["response"])
            else:
                r_array.append("")
        except:
            r_array.append("")
    
    # Instructions
    code = ''.join(r_array)
    #idx_code = code.find('<code>')
    #after_idx = idx_code+len('<code>')
    #code = code[after_idx:]
    return code, duration

def run_ollama(model, temperature, idx, test, completed_set):
    def get_completed_ids(model):
        """
            Return list of generated problems' ids
            and next index file
        """
        problems_ids = set([])
        last_idx = []
        folder_path = 'data/'
        for filename in os.listdir(folder_path):
            if f'{model}-grammar' in filename:
                file_path = os.path.join(folder_path, filename)
                # Search for the sequence of digits in the string
                match = re.search(r'\d+', filename)
                # Extract the matched digits as a string
                if match:
                    number = match.group()
                    last_idx.append(int(number))
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # Parse each line as a JSON object
                        json_obj = json.loads(line.strip())
                        if json_obj['code'][0] != "timed out":
                        	problems_ids.add(json_obj['metadata']['problem_id'])
        return problems_ids, str(max(last_idx)+1)


    def get_grammar(library_name="Pandas"):
        """
            Reads a file containing a GBNF grammar
            returns it as a string
        """
        file_name = f"../../grammar/files/{library_name.lower()}-reorder-grammar.gbnf"

        with open(file_name, 'r') as file:
            content = file.read()
        return content

    def run_func(prompt, id, model_sample="", grammar="",timeout=60):
        code, duration = call_ollama_api(
                        # 'Write short code following the problem format and indentation. Do not include formatting tags.',
            '',
            prompt,
            grammar=grammar,
            temperature=temperature,
            n=1,
            model=model,
            max_tokens=1024,
            timeout=60
        )
        return {'task_id': id, 'solution': code, 'model': model_sample, 'duration': duration}

    #completed_ids = set({})
    file_output = f'results/{model}-repair-{test}-{idx}-answers.jsonl'
    with cfuts.ThreadPoolExecutor(
        max_workers=4
    ) as executor:
        futs = []
        for p in bigcode_repair: 
            id = p['id']
            model_sample = p['model']
            if (id, model_sample) not in completed_set:
                prompt = "Repair the following code:\n"
                if test == "union":
                    prompt =  prompt + p['code']
                    prompt = prompt + "\n Using this analysis: \n"
                    if "line -1" in p['static_union_str']:
                        prompt = prompt + "Syntax error. \n"
                    else:
                        prompt = prompt + p['static_union_str']
                    if p['static_union_str'] != "":
                        print(prompt)
                         # you have some options on how to actually execute the program here.
                        futs.append(executor.submit(run_func, prompt, id, model_sample))
                elif test == "mypy":
                    prompt =  prompt + p['code']
                    prompt = prompt + "\n Using this analysis: \n"
                    if "line -1" in p['mypy_str']:
                        prompt = prompt + "Syntax error. \n"
                    else:
                        prompt = prompt + p['mypy_str']
                    if p['mypy_str'] != "":
                        print(prompt)
                         # you have some options on how to actually execute the program here.
                        futs.append(executor.submit(run_func, prompt, id, model_sample))
                    else:
                        continue
                elif test == "pyright":
                    prompt =  prompt + p['code']
                    prompt = prompt + "\nUsing this analysis: \n"
                    if "line -1" in p['pyright_str']:
                        prompt = prompt + "Syntax error. \n"
                    else:
                        prompt = prompt + p['pyright_str']
                    if p['pyright_str'] != "":
                        print(prompt)
                         # you have some options on how to actually execute the program here.
                        futs.append(executor.submit(run_func, prompt, id, model_sample))
                elif test == "general":
                    prompt = prompt + p['code']
                    # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id, model_sample))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            with open(file_output, 'a') as f:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    """ Example """
    """ python run_ollama_repair.py --test "union" """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="granite-code:latest",
        help="which model to run")
    parser.add_argument(
        "--idx",
        type=str,
        default="0",
        help="index of sample"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="union",
        help="repair analysis to include"
    )

    args = parser.parse_args()
    bigcode_repair = [json.loads(l) for l in open("results/bigcodebench_error_repair.jsonl", "r")]
    file_output = f'results/{args.model}-repair-{args.test}-{args.idx}-answers.jsonl'
    completed_set = set()
    if os.path.exists(file_output):
        completed_set = get_completed_ids_file(file_output)
    print(completed_set)
    temp = 0.2
    run_ollama(args.model, temp, args.idx, args.test, completed_set)