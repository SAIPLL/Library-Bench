"""Evaluate LLM performance on NL-to-Code generation, using concurrent threads. """

import os, random  
import time, json, argparse
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

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

client_OA = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

odex_mappings = [json.loads(l) for l in open("odex_grammar_mapping.jsonl", "r")]

def get_grammar_file(library_lst):
    lib_str = ','.join(sorted(library_lst))
    for m in odex_mappings:
        if m["library"] == lib_str:
            return f'../../grammar/{m["grammar"]}'
    return ""
    return

def get_grammar(file_name="question1-grammar.gbnf"):
    """
        Reads a file containing a GBNF grammar
        returns it as a string
    """
    with open(file_name, 'r') as file:
        content = file.read()
    return content

def get_response_claude(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False,
    grammar: bool = False, 
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    prompt = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    response = client.messages.create(
        model=args.model_name, 
        messages=prompt,
        max_tokens=args.max_tokens, 
        temperature=args.temperature
    )
    return [response.content[0].text]

def get_response_gpt(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
    grammar: bool = False,
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    prompt = [
        {"role": "user", "content": prompt}
    ]

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


def get_response_ollama_api(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
    grammar: bool = False,
) -> List[str]:
    # Add suffix if exists
    if sample["suffix"] != "":
        prompt = prompt+sample["suffix"]
    try:
        response = ollama.generate(
            model=args.model_name,
            prompt=prompt,
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

def get_response_ollama(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
    grammar: bool = False,
    timeout: int = 60,
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    # Get call
    payload = {
        "model": args.model_name,
        "temperature": args.temperature,
        "prompt": ''.join(prompt),
        "stream": True,
        "options": {},
        "think": False,
        "keep_alive": -1
    }
            
    if args.top_p > 0:
        payload["options"]["top_p"] = args.top_p

    payload["options"]["num_ctx"] = 2048

    if grammar:
        # Get sample task_id
        # TODO: Modify path
        grammar = get_grammar_file(sample['library'])
    else:
        grammar = ""

    if len(grammar) > 0:
       payload["grammar"] = get_grammar(grammar)
    
    headers = {
        "Content-Type": "application/json"
    }
    
    url = "http://localhost:11434/api/generate"

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)

        # Split the string into individual JSON objects
        json_objects = response.text.strip().split('\n')
        # Parse each JSON object
        parsed_objects = [json.loads(obj) for obj in json_objects]
        r_array = []
        for i in parsed_objects:
            r_array.append(i["response"])
        
        # Instructions
        code = ''.join(r_array)

        return[code]
    except requests.exceptions.Timeout:
        return ["timed out"]
    return ["other error"]


RESPOND_DICT = {
    "gpt-4-0613": get_response_gpt,
    "gpt-3.5-turbo-0125": get_response_gpt,
    "claude-3-opus-20240229": get_response_claude,
    "granite-code:latest": get_response_ollama,
    "gemma2:2b": get_response_ollama,
    "gemma2:9b": get_response_ollama,
    "qwen3:8b": get_response_ollama,
    "claude-haiku-4-5-20251001": get_response_claude,
    "claude-sonnet-4-6": get_response_claude
}

def get_predictions(
    model_name: str, prompt: str, 
    sample: Dict, index: int, 
    sleep_time: int, verbose: bool,
    grammar: bool 
) -> List[str]:

    if sleep_time == 0: 
        return RESPOND_DICT[model_name](prompt, sample, verbose, grammar)
    
    # enable sleep time otherwise
    predictions = None 
    while predictions is None: 
        try: 
            predictions = RESPOND_DICT[model_name](prompt, sample, verbose, grammar)
            print(f"predicted at sample #{index}")
        except: 
            print(f"sleep for {sleep_time} at sample #{index}")
            time.sleep(sleep_time)
    return predictions


def select_fewshot_examples(
    sample: Dict, candidates: List[Dict], 
    num_examples: int = 1, method: str = "random", 
) -> List[Dict]: 
    """Select example as prefix to the prompt of the current sample. """
    if method == "random": 
        num_examples = min(num_examples, len(candidates))
        return random.sample(candidates, num_examples)

def process_sample(i: int, sample: Dict, dataset: List[Dict]) -> None:
    """Process a single sample: build prompt, get predictions, write result."""
    # Just open-domain questions

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

    predictions = get_predictions(
        model_name=args.model_name,
        prompt=prompt, sample=sample, index=i,
        sleep_time=args.sleep_time, verbose=args.verbose,
        grammar=eval(args.grammar)
    )

    print(f"predicted: {predictions[0]} \n")

    num_valid = len(predictions)
    assert num_valid == args.n, f"# num_valid"

    result_pred = {
        "predictions": predictions[0],
        "task_id": sample["task_id"],
        "library": sample["library"]
    }

    return result_pred


def main():
    dataset = load_testset(args.input_path)
    file_output = args.output_path
    file_output = file_output.replace('.json', '.jsonl')
    print(f'Output file: {file_output}')

    args.verbose = False

    with cfuts.ThreadPoolExecutor(max_workers=6) as executor:
        futs = []

        for i, sample in enumerate(dataset):
            if "import " in sample["test_start"]:
                futs.append(executor.submit(process_sample, i, sample, dataset))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            with open(file_output, 'a') as f:
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=str, default="False")
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)

    parser.add_argument("--model_name", type=str, default="code-davinci-002", 
        choices=["code-cushman-001", "code-davinci-001", "code-davinci-002", "gpt-4-0613", "claude-3-opus-20240229", "gpt-3.5-turbo-0125", "granite-code:latest", "gemma2:2b", "gemma2:9b", "qwen3:8b", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1, 
        help="Number of predictions required for each api call.")
    
    parser.add_argument("--sleep_time", type=int, default=30, 
        help="Specify a positive integer if enable time sleep.")
    
    parser.add_argument("--function_name", type=str, default="id", 
        choices=["id", "constant", "intent"], 
        help="Method to construct the function name. ")
    parser.add_argument("--num_examples", type=int, default=0, 
        help="Number of examples included in the current prompt input. ")
    parser.add_argument("--fewshot_method", type=str, default="random", 
        choices=["random"], 
        help="Method to select the prefix examples for prompt creation.")
    parser.add_argument("--strip_prompt", action="store_true",
        help="Whether to strip the trailing whitespaces in the prompt. ")

    #parser.add_argument("--openai_api_key", type=str, default=None)
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
            raise Exception(f"Need to specify [language] or [i/o path]")
        if not args.input_path: 
            args.input_path = get_test_path(args.language)
        if not args.output_path: 
            args.output_path = get_prediction_path(
                args.model_name, args.language, 
                args.num_examples, args.num_tests, 
            )
            args.output_path = args.output_path.replace(':','-')
            if eval(args.grammar):
                args.output_path = args.output_path.replace('.json','-grammar.json')
            print(args.output_path)

    main()