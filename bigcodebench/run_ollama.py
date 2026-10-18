import json
import os
import sys
import argparse
import requests
import concurrent.futures as cfuts
import time
import re
import pandas as pd
import ollama
from anthropic import Anthropic

from tqdm import tqdm

client_anthropic = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def get_dataset_hf():
    splits = {'v0.1.0_hf': 'data/v0.1.0_hf-00000-of-00001.parquet', 'v0.1.1': 'data/v0.1.1-00000-of-00001.parquet', 'v0.1.2': 'data/v0.1.2-00000-of-00001.parquet', 'v0.1.3': 'data/v0.1.3-00000-of-00001.parquet', 'v0.1.4': 'data/v0.1.4-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/bigcode/bigcodebench/" + splits["v0.1.0_hf"]) 
    return df


def call_anthropic_api(system_prompt, prompt, temperature, n, model, max_tokens) -> list[str]:
    prompt = [
        {"role": "user", "content": [{"type": "text", "text":prompt}]}
    ]
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
            import time; time.sleep(10); pass
    # Handle non-successful stop reasons
    stop_reason = result.stop_reason
    if stop_reason not in ("end_turn", "stop_sequence", "max_tokens"):
        return [f"ERROR:{stop_reason}"]

    # Guard against empty content
    if not result.content:
        return ["ERROR:empty_response"]
    
    code = result.content[0].text
    idx_code = code.find('<code>')
    after_idx = idx_code+len('<code>')
    code = code[after_idx:]
    return [code]

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
        if c['solution'][0] != 'timed out':
            completed_set.add(c['task_id'])
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
                "stream": False,
                "options": {},
                "think": False,
                "keep_alive": keep_alive,
            }
            payload["options"]["num_ctx"] = 2048
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
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=120)
            break
        except requests.exceptions.Timeout:
            time_end = time.time()
            duration = time_end - time_start
            time.sleep(10);
            #if x < attempts:
                #continue
            #else:
            return ["timed out"], duration

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

def call_ollama_api_api(system="", prompt="something", model="gemma2:2b", temperature=1.0):
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            system=system,
            stream=False,
            keep_alive="6h",
            think=False,
            options = {
                "num_ctx": 2048, 
            }
        )
        
        full_text = response['response']
        
        return [full_text]

    except Exception as e:
        return [f"Error: {str(e)}"]

def run_ollama(model, temperature, idx, completed_set, bigcodebench_mappings, grammar):
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

    def find_grammar_bigcodebench(bigcodebench_mappings, library_name):
        lib_str = library_name.replace('[','').replace(']','').replace(' ','').replace('\'','')
        lib_str = ','.join(sorted(lib_str.split(',')))
        
        if lib_str in bigcodebench_mappings:
            id = bigcodebench_mappings[lib_str]['id']
            file_grammar = f'../../grammar/files/bigcodebench/{id}-atomimport-grammar.gbnf'
            
            with open(file_grammar, 'r') as file:
                content = file.read()
            return content
        
        return ""

    def run_func2(prompt, id, model_sample="", grammar_st="",timeout=60):
        code, duration = call_ollama_api(
                        # 'Write short code following the problem format and indentation. Do not include formatting tags.',
            '',
            prompt,
            grammar=grammar_st,
            temperature=temperature,
            n=1,
            model=model,
            max_tokens=1024,
            timeout=timeout
        )
        return {'task_id': id, 'solution': code, 'duration': duration}

    def run_func(prompt, id, model=model,timeout=60):
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
            code = call_ollama_api_api(
                            # 'Write short code following the problem format and indentation. Do not include formatting tags.',
                '',
                prompt,
                model=model,
                temperature=temperature,
            )
        return {'task_id': id, 'solution': code}

    #completed_ids = set({})
    #file_output = f'results/{model}-{idx}-answers.jsonl'
    with cfuts.ThreadPoolExecutor(
        max_workers=5
    ) as executor:
        futs = []
        for i, task_info in bigcodebench.iterrows():
            if i >= 0 and i < 1200:
                id = task_info['task_id']
                if id not in completed_set:
                    prompt = task_info['instruct_prompt']
                    print(prompt)
                else:
                    continue
                library_name = task_info['libs']
                if eval(grammar):
                    grammar_st = find_grammar_bigcodebench(bigcodebench_mappings, library_name)
                futs.append(executor.submit(run_func2, prompt, id, grammar_st=grammar_st))
                #futs.append(executor.submit(run_func, prompt, id))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            with open(file_output, 'a') as f:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    """ Example """
    """ python run_ollama.py --model "gemma2:2b" """
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
        "--grammar",
        type=str,
        default="False",
        help="index of sample"
    )


    args = parser.parse_args()
    bigcodebench = get_dataset_hf()
    with open("bigcodebench_mapping.json", "r") as f:
        bigcodebench_mappings = json.load(f)
        f.close()
    file_output = f'results/{args.model}-{args.idx}-no-constrained-answers.jsonl'
    completed_set = set()
    if os.path.exists(file_output):
        completed_set = get_completed_ids_file(file_output)
    print(completed_set)
    print(file_output)
    temp = 1
    run_ollama(args.model, temp, args.idx, completed_set, bigcodebench_mappings, args.grammar)