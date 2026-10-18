import json
import os
import sys
import gzip
import argparse
import requests
import concurrent.futures as cfuts
import time
import re
import ollama

from tqdm import tqdm

def call_ollama_api_api(system="", prompt="something", model="gemma2:2b", temperature=0.2):
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

        # Extracción rápida del código
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
    """
        Call the model with either stop token,
        or n look-ahead tokens
    """
    while True:
        time_dct = {
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0
        }
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
            payload["options"]["num_ctx"] = 2048
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
            return ["timed out"]#, time_dct
    
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
    
    # Instructions
    code = ''.join(r_array)
    #idx_code = code.find('<code>')
    #after_idx = idx_code+len('<code>')
    #code = code[after_idx:]
    #return [code]
    return [code]#, time_dct


def run_ollama(model, temperature, idx, with_grammar, target_library, l_i, u_i):
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
        if len(last_idx) > 1:
            max_idx = str(max(last_idx)+1)
        else:
            max_idx = "0"

        return problems_ids, max_idx


    def get_grammar(library_name="Pandas"):
        """
            Reads a file containing a GBNF grammar
            returns it as a string
        """
        file_name = f"../../grammar/files/{library_name.lower()}-atomimport-grammar.gbnf"

        with open(file_name, 'r') as file:
            content = file.read()
        return content

    def run_func(prompt, id, grammar, timeout=60, metadata=None):
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
        #code = call_ollama_api_api(system='Write a short code following the given format and indentation. Place the executable code between <code> '
            #'and </code> tags, without any other non-executable things.', prompt=prompt, model=model, temperature=temperature)
        """func_dct = {'id': id, 
                    'code': code, 
                    'metadata': metadata,
                    'total_duration': time_d['total_duration'],
                    'load_duration': time_d['load_duration'],
                    'prompt_eval_count': time_d['prompt_eval_count'],
                    'prompt_eval_duration': time_d['prompt_eval_duration'],
                    'eval_count': time_d['eval_count'],
                    'eval_duration': time_d['eval_duration']
                    }"""
        """func_dct = {'id': id, 
                    'code': code, 
                    'metadata': ds1000[id]['metadata']}"""   
        func_dct = {'id': id, 
                    'code': code, 
                    'metadata': metadata}
        return func_dct

    ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()][l_i:u_i]
    ds1000_responses = []
    completed_ids = set({})
    if with_grammar:
        completed_ids, last_idx = get_completed_ids(model)
        #if type(last_idx) == str:
            #idx = last_idx
        print(completed_ids)
    #print(idx)
    if with_grammar:
    	file_output = f'data/{model}-grammar-{idx}-answers.jsonl'
    else:
    	file_output = f'data/{model}-no-grammar-{idx}-answers.jsonl'
    print(f'File output: {file_output}')
    with cfuts.ThreadPoolExecutor(
        max_workers=6
    ) as executor:
        futs = []
        for p in ds1000:
            id = int(p['metadata']['problem_id'])
            if id not in completed_ids:
                #print(id)
                prompt = p['prompt']
                library_needed = p['metadata']['library']
                if (library_needed.lower() == target_library) or target_library == "":
                    if with_grammar:
                        library_grammar = get_grammar(library_needed)
                        print(f'Prompt #{id}')
                    else:
                    	library_grammar = ""
                    # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id, library_grammar, timeout=240, metadata=p['metadata']))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            with open(file_output, 'a') as f:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    """ Example """
    """ python run_ollama.py --grammar "False" --library "" """
    """ python run_ollama.py --grammar "True" --model='gemma2:2b' """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="granite-code:latest",
        help="which model to run")
    parser.add_argument(
        "--idx",
        type=str,
        default="1",
        help="index of sample"
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default="False",
        help="to call ollama with grammar"
    )
    parser.add_argument(
    	"--library",
    	type=str,
    	default="",
    	help="to query DS1000 in a specific library"
    )

    args = parser.parse_args()
    temp = 1
    #run_ollama(args.model, temp, args.idx, eval(args.grammar), args.library, lower_index, upper_index)
    run_ollama(args.model, temp, args.idx, eval(args.grammar), args.library, 0, 10)