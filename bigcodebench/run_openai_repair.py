import json
import os
import sys
import gzip
import argparse
import concurrent.futures as cfuts
import time

from tqdm import tqdm
from openai import OpenAI,OpenAIError

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def call_openai_api(system_prompt, prompt, temperature, n, model, max_tokens, stop) -> list[str]:
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    while True:
        time_taken = 0
        try:
            start = time.time()
            result = client.chat.completions.create(
                model='o3-mini',
                messages=prompt,
                temperature=temperature,
                n=n,
                stop=stop
            )
            end = time.time()
            time_taken = end-start
            break
        except OpenAIError as e:
            # Handle all OpenAI API errors
            print(f"Error: {e}")
    return [result.choices[i].message.content for i in range(n)], time_taken


def run_openai(model, temperature, idx, test):
    def run_func(prompt, id,  model_sample):
        code, time_taken = call_openai_api(
                        # 'Write short code following the problem format and indentation. Do not include formatting tags.',
            'Write the code with modifications, do not write explanations',
            prompt,
            temperature=temperature,
            n=1,
            model='o3-mini',
            max_tokens=1024,
            stop=["</code>", "# SOLUTION END"],
        )
        #code = [""]
        #time_taken = 0
        return {'task_id': id, 'solution': code[0], 'model':  model_sample, 'duration':time_taken, 'prompt': prompt}

    responses = []
    test = "self"
    file_output = f'results/{model}-repair-{test}-{idx}-answers.jsonl'
    print(file_output)
    with cfuts.ThreadPoolExecutor(
        max_workers=32
    ) as executor:
        futs = []
        for p in bigcode_repair:
            id = f"BigCodeBench/{p['cid']}"
            model_sample = model
            prompt = "Repair the following code:\n"
            if test == "union":
                prompt =  prompt + p['lines']
                prompt = prompt + "\n Using this analysis: \n"
                if "line -1" in p['static_union_str']:
                    prompt = prompt + "Syntax error. \n"
                else:
                    rompt = prompt + p['static_union_str']
                if p['static_union_str'] != "":
                    print(prompt)
                     # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id, model_sample))
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
                    futs.append(executor.submit(run_func, prompt, id, model_sample))
            elif test == "pyright":
                prompt =  prompt + p['lines']
                prompt = prompt + "\n Using this analysis: \n"
                if "line -1" in p['pyright_str']:
                    prompt = prompt + "Syntax error. \n"
                else:
                    prompt = prompt + p['pyright_str']
                if p['pyright_str'] != "":
                    print(prompt)
                     # you have some options on how to actually execute the program here.
                    futs.append(executor.submit(run_func, prompt, id, model_sample))
            else:
                prompt = prompt + p['lines']
                # you have some options on how to actually execute the program here.
                futs.append(executor.submit(run_func, prompt, id, model_sample))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            responses.append(result)

    with open(file_output, 'w') as f:
        for r in responses:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    """ python run_openai_repair.py --model='gemma2-2b' --test='pyright' """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma2-2b",
        help="which results to run")
    parser.add_argument(
        "--idx",
        type=str,
        default="0",
        help="index of sample"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="mypy",
        help="repair analysis to include"
    )
    args = parser.parse_args()
    read_file = f"results/{args.model}-repair-{args.test}-{args.idx}-questions.jsonl"
    print(read_file)
    bigcode_repair = [json.loads(l) for l in open(read_file, "r")]
    temp = 1
    run_openai(args.model, temp, args.idx, args.test)