"""Evaluate Codex performance on NL-to-Code generation. """

import os, random  
import time, json, argparse
from typing import Dict, List  
from src.utils import get_test_path, get_prediction_path, load_testset
from prompt import create_fewshot_prompt_nl2code
from verify import get_valid_solutions, wrap_check

import requests
import anthropic
from openai import OpenAI

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

client_OA = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_grammar(file_name="question1-grammar.gbnf"):
    """
        Reads a file containing a GBNF grammar
        returns it as a string
    """
    with open(file_name, 'r') as file:
        content = file.read()
    return content

def get_response_001(
    prompt: str, 
    verbose: bool = False,
    gramar: bool = False,
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    response = openai.Completion.create(
        model=args.model_name, 
        prompt=prompt, 
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        n=args.n, 
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"], 
    )
    return [choice["text"] for choice in response["choices"]]


def get_response_davinci_002(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
    grammar: bool = False,
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    response = openai.Completion.create(
        model=args.model_name, 
        prompt=prompt, 
        suffix=sample["suffix"], 
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        n=args.n, 
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return [choice["text"] for choice in response["choices"]]

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
        temperature=args.temperature, 
        top_p=args.top_p
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

def get_response_ollama(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
    grammar: bool = False,
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    # Get call
    payload = {
        "model": args.model_name,
        "temperature": args.temperature,
        "prompt": ''.join(prompt),
        "stream": True,
        "options": {},
        "keep_alive": -1
    }
            
    if args.top_p > 0:
        payload["options"]["top_p"] = args.top_p

    if grammar:
        # Get sample task_id
        # TODO: Modify path
        grammar = f"../../grammar/files/odex/{args.language}_{sample['task_id']}-reorder-grammar.gbnf"
    else:
        grammar = ""

    if len(grammar) > 0:
       payload["grammar"] = get_grammar(grammar)
    
    headers = {
        "Content-Type": "application/json"
    }
    
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # Split the string into individual JSON objects
    json_objects = response.text.strip().split('\n')
    # Parse each JSON object
    parsed_objects = [json.loads(obj) for obj in json_objects]
    r_array = []
    for i in parsed_objects:
        r_array.append(i["response"])
    
    # Instructions
    code = ''.join(r_array)

    return [code]

RESPOND_DICT = {
    "code-cushman-001": get_response_001, 
    "code-davinci-001": get_response_001, 
    "code-davinci-002": get_response_davinci_002,
    "gpt-4-0613": get_response_gpt,
    "gpt-3.5-turbo-0125": get_response_gpt,
    "o3-mini": get_response_gpt,
    "claude-3-opus-20240229": get_response_claude,
    "granite-code:latest": get_response_ollama
}

def get_predictions(
    model_name: str, prompt: str, 
    sample: Dict, index: int, 
    sleep_time: int, verbose: bool,
    grammar: bool 
) -> List[str]:
    print(f"prompt: {prompt} \n")

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

def get_prompt_repair(p, repair_type, prompt_original):
    if p["lines"] == "":
        return ""
    prompt = "Repair the following code:\n"
    if "def f_" not in p["lines"]:
        lines = prompt_original + '\n' + p["lines"]
    else:
        lines = p["lines"]
    if repair_type == "union":
        prompt =  prompt + lines
        prompt = prompt + "\n Using this analysis: \n"
        if "line -1" in p['static_union_str']:
            prompt = prompt + "Syntax error. \n"
        else:
            prompt = prompt + p['static_union_str']
        if p['static_union_str'] == "":
            return ""
    elif repair_type == "mypy":
        prompt =  prompt + lines
        prompt = prompt + "\n Using this analysis: \n"
        if "line -1" in p['mypy_str']:
            prompt = prompt + "Syntax error. \n"
        else:
            prompt = prompt + p['mypy_str']
        if p['mypy_str'] == "":
            return ""
    elif repair_type == "pyright":
        prompt =  prompt + lines
        prompt = prompt + "\nUsing this analysis: \n"
        if "line -1" in p['pyright_str']:
            prompt = prompt + "Syntax error. \n"
        else:
            prompt = prompt + p['pyright_str']
        if p['pyright_str'] == "":
            return ""
    elif repair_type == "general":
        prompt = prompt + lines
    return prompt

def get_id_number(file_completion):
    file_completion = file_completion.replace('.py','')
    idx = file_completion.find('_')
    file_number = file_completion[idx+1:]
    idx2 =  file_number.find('_')
    if idx2 > -1:
        return int(file_number[:idx2])
    else:
        return int(file_number)

def find_get_repair(odex_repair, task_id, repair_type, context_prompt):
    prompts = []
    for e in odex_repair:
        if get_id_number(e["completion_id"]) == task_id:
            prompt = get_prompt_repair(e, repair_type, context_prompt)
            if prompt != "":
                prompts.append(prompt)
    return len(prompts) > 0, prompts

def main(): 
    # load source dataset
    dataset = load_testset(args.input_path)
    odex_repair = [json.loads(l) for l in open("data/odex_repair.jsonl", "r")]
    
    predset = [] 
    scores_dict = {f"pass@{idx}": [] for idx in range(1, args.n+1)}

    args.verbose = True

    for i, sample in enumerate(dataset):
        # Just open-domain questions
        # these means import to a library
        if "import " not in sample["test_start"]:
            continue
        # Find if record is needed for repair 
        prompt_original = ""
        if "prompt" in sample:
            prompt_original = sample["prompt"]
        is_repair, prompts = find_get_repair(odex_repair, sample["task_id"], args.repair_type, prompt_original)
        if not is_repair:
            print("Next")
        else:
            for prompt in prompts:
                # create model input -- prompt 
                examples = select_fewshot_examples(
                    sample=sample, 
                    candidates=dataset[:i]+dataset[i+1:], 
                    num_examples=args.num_examples, 
                    method=args.fewshot_method,
                )

                if args.strip_prompt:
                    prompt = prompt.rstrip()

                # collect code predictions 
                predictions = get_predictions(
                    model_name=args.model_name, 
                    prompt=prompt, sample=sample, index=i, 
                    sleep_time=args.sleep_time, verbose=args.verbose,
                    grammar=eval(args.grammar)
                )

                print(f"predicted: {predictions[0]} \n")
                
                # simple cleansing of predicions
                valid_predictions = get_valid_solutions(predictions, deduplicate=False)
                #print(f"valid: {valid_predictions[0]} \n")
                num_valid = len(predictions)
                assert num_valid == args.n, f"# num_valid"
                
                valid_pred = [valid_predictions[0]]

                print(f"valid: {valid_pred[0]} \n")
                
                scores, outputs = wrap_check(
                    sample, valid_pred, 
                    k=[i+1 for i in range(num_valid)], 
                    num_workers=args.n, 
                    max_num_tests=args.num_tests_eval, 
                    verbose=args.verbose, 
                    function_name=args.function_name, 
                )
                print(outputs)
                #if i % 10 == 0: 
                print(f"[scores@{i:3d}] {scores}")

                for idx in range(num_valid): 
                    key = f"pass@{idx+1}"
                    if key in scores: 
                        scores_dict[key].append(scores[key])
                predset.append({
                    "output": outputs, 
                    "predictions": predictions[0], 
                    "task_id": sample["task_id"]
                })
    
    # write records to prediction file 
    json.dump(predset, open(args.output_path, 'w'))
    
    for idx in range(args.n): 
        key = f"pass@{idx+1}"
        scores = scores_dict[key]
        print(f"[{key}] {sum(scores)/len(scores):.3f} ({len(scores)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=str, default="False")
    parser.add_argument("--repair_type", type=str, default="union")
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)

    parser.add_argument("--model_name", type=str, default="code-davinci-002", 
        choices=["code-cushman-001", "code-davinci-001", "code-davinci-002", "gpt-4-0613", "claude-3-opus-20240229", "gpt-3.5-turbo-0125", "granite-code:latest", "o3-mini"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
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
            args.output_path = args.output_path.replace('.json',f'-{args.repair_type}.json')
            print(args.output_path)

    main()