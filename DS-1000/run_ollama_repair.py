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

def call_ollama_api(system="", prompt="something", 
                    url="http://localhost:11434/api/generate", 
                    grammar="", n_probs=0, top_p=0, temperature=0.2, 
                    model="granite-code:latest", keep_alive=-1,
                    stop=[], max_tokens=1024, n=1, timeout=60, attempts=3):
    """
        Call the model with either stop token,
        or n look-ahead tokens
    """
    while True:
        try:
            # Get call
            payload = {
                "model": model,
                "temperature": temperature,
                "prompt": ''.join(prompt),
                "stream": True,
                "options": {},
                "system": system,
                "keep_alive": keep_alive
            }
            
            if n_probs > 0:
                payload["options"]["n_probs"] = n_probs
            if top_p > 0:
                payload["options"]["top_p"] = top_p
            if len(grammar) > 0:
                payload["grammar"] = grammar
            
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
    return [code]

def run_ollama(model, temperature, idx, test):
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

    def run_func(prompt, id, grammar="", timeout=60):
        code = call_ollama_api(
                        # 'Write short code following the problem format and indentation. Do not include formatting tags.',
            'Write a short code following the given format and indentation. Place the executable code between <code> '
            'and </code> tags, without any other non-executable things.',
            prompt,
            grammar=grammar,
            temperature=temperature,
            n=1,
            model=model,
            max_tokens=1024,
            stop=["</code>", "# SOLUTION END"],
            timeout=60
        )
        return {'id': id, 'code': code, 'metadata': ds1000[id]['metadata']}

    ds1000_responses = []
    completed_ids = set({})
    file_output = f'data/{model}-repair-{test}-{idx}-answers.jsonl'
    with cfuts.ThreadPoolExecutor(
        max_workers=6
    ) as executor:
        futs = []
        for p in ds_repair:
            id = int(p['completion_id'])
            prompt = "Repair the following code:\n"
            if test == "union":
                prompt =  prompt + p['lines']
                prompt = prompt + "\n Using this analysis: \n"
                if "line -1" in p['static_union_str']:
                    prompt = prompt + "Syntax error. \n"
                else:
                    prompt = prompt + p['static_union_str']
                if p['static_union_str'] != "":
                    print(prompt)
                     # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id))
            elif test == "mypy":
                prompt =  prompt + p['lines']
                prompt = prompt + "\n Using this analysis: \n"
                if "line -1" in p['mypy_str']:
                    prompt = prompt + "Syntax error. \n"
                else:
                    prompt = prompt + p['mypy_str']
                if p['mypy_str'] != "":
                    print(prompt)
                     # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id))
                else:
                    continue
            elif test == "pyright":
                prompt =  prompt + p['lines']
                prompt = prompt + "\nUsing this analysis: \n"
                if "line -1" in p['pyright_str']:
                    prompt = prompt + "Syntax error. \n"
                else:
                    prompt = prompt + p['pyright_str']
                if p['pyright_str'] != "":
                    print(prompt)
                     # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id))
            elif test == "general":
                prompt = prompt + p['lines']
                # you have some options on how to actually execute the program here.
                futs.append(executor.submit(run_func, prompt, id))

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

    ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]
    ds_repair = [json.loads(l) for l in open("data/ds_repair.jsonl", "r")]
    args = parser.parse_args()
    temp = 0.2
    run_ollama(args.model, temp, args.idx, args.test)