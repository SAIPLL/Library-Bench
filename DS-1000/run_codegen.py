import json
import os
import re
import sys
import gzip
import argparse
import concurrent.futures as cfuts


from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from torch import LongTensor, FloatTensor, eq
import torch

def initialize_model(model):
    """
        Loads the model and returns stopping criteria
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    stop_list = ["END SOLUTION"]
    stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
    stop_token_ids = [LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if (input_ids[0][-len(stop_ids[0])+1:] == stop_ids[0][1:]).all():
                    return True
            return False

    stopping_criteria = StopOnTokens()

    return tokenizer, model, stopping_criteria

def extract_solution(content):
    pattern = r"BEGIN SOLUTION(.*?)END SOLUTION"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

def generate_codegen(system_prompt, prompt, n, max_tokens, stop, tokenizer, model) -> list[str]:
    text = f"{system_prompt}\n{prompt}"
    print(text)
    return
    while True:
        try:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_length=max_tokens, stopping_criteria=[stop])
            code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            break
        except:
            import time; time.sleep(10); pass
    if not code:
        code = ""
    else:
        code = extract_solution(code)
    return [code]


def run_codegen(model, idx, tokenizer, model_object, stopping_criteria):
    def run_func(prompt, id):
        code = generate_codegen(
                        # 'Write short code following the problem format and indentation. Do not include formatting tags.',
            'Write a short code following the given format and indentation. Place the executable code between <code> '
            'and </code> tags, without any other non-executable things.',
            prompt,
            n=1,
            max_tokens=1024,
            stop=stopping_criteria,
            tokenizer=tokenizer,
            model=model_object
        )
        return {'id': id, 'code': code, 'metadata': ds1000[id]['metadata']}

    ds1000_responses = []
    with cfuts.ThreadPoolExecutor(
        max_workers=32
    ) as executor:
        futs = []
        for p in ds1000:
            id = int(p['metadata']['problem_id'])
            prompt = p['prompt']
            # you have some options on how to actually execute the program here.
            futs.append(executor.submit(run_func, prompt, id))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            ds1000_responses.append(result)
    ds1000_responses.sort(key=lambda x: int(x['id']))
    model_name = model.split('/')
    with open(f'data/{model_name[1]}-{idx}-answers.jsonl', 'w') as f:
        for r in ds1000_responses:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/codegen-350M-multi",
        help="which results to run")
    parser.add_argument(
        "--idx",
        type=str,
        default="0",
        help="index of sample"
    )

    ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]
    args = parser.parse_args()
    tokenizer, model, stopping_criteria = initialize_model(args.model)
    run_codegen(args.model, args.idx, tokenizer, model, stopping_criteria)
    # run_openai('gpt-3.5-turbo-0125', 0)
    # run_openai('gpt-4-0613', 0)
    # run_openai('gpt-4-turbo-2024-04-09', 0)