import json
import os
import sys
import gzip
import argparse
import concurrent.futures as cfuts

from tqdm import tqdm
from anthropic import Anthropic
from openai import OpenAI
from google import genai

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

client_open = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

client_gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

def call_anthropic_api(system_prompt, prompt, temperature, n, model, max_tokens, stop) -> list[str]:
    prompt = [
        {"role": "user", "content": [{"type": "text", "text":prompt}]}
    ]
    while True:
        try:
            result = client.messages.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system_prompt,
                stop_sequences=stop
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

def call_openai_api(system_prompt, prompt, temperature, model, stop) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            result = client_open.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                reasoning_effort="none"
            )
            break
        except Exception as e:
            print(e)
            import time; time.sleep(10)

    finish_reason = result.choices[0].finish_reason
    if finish_reason not in ("stop", "length"):
        return f"ERROR:{finish_reason}"

    content = result.choices[0].message.content
    if not content:
        return "ERROR:empty_response"

    idx_code = content.find('<code>')
    if idx_code == -1:
        return content

    after_idx = idx_code + len('<code>')
    return content[after_idx:]

def call_gemini_api(system_prompt, prompt, temperature, model, stop) -> str:
    while True:
        try:
            interaction = client_gemini.interactions.create(
                model=model,  # e.g. "gemini-3.5-flash"
                system_instruction=system_prompt,
                input=prompt,
                store=False,  # <-- stateless: don't persist this interaction server-side
                generation_config={
                    "temperature": temperature,
                    #"max_output_tokens": max_tokens,
                    "stop_sequences": stop,
                    "thinking_level": "minimal",  # closest analog to reasoning_effort="none"; use "none" if your model/tier supports it
                }
            )
            break
        except Exception as e:
            import time; time.sleep(10)

    text = interaction.output_text
    if not text:
        return "ERROR:empty_response"

    idx_code = text.find('<code>')
    if idx_code == -1:
        return text

    after_idx = idx_code + len('<code>')
    return text[after_idx:]

def run_claude(model, temperature, idx):
    def run_func(prompt_text, id):
        s_prompt = 'Write a short code following the given format and indentation. Place the executable code between <code> and </code> tags, without any other non-executable things.'
        if 'claude' in model:
            code = call_anthropic_api(
                system_prompt=s_prompt,
                prompt=prompt_text,
                temperature=temperature,
                n=1,
                model=model,
                max_tokens=4096,
                stop=["</code>", "# SOLUTION END"],
            )
        elif 'gemini' in model:
            code = call_gemini_api(
                system_prompt=s_prompt,
                prompt=prompt_text,
                temperature=temperature,
                model=model,
                #max_tokens=2048,
                stop=["</code>", "# SOLUTION END"],
            )
        else:
            code = call_openai_api(
                system_prompt=s_prompt,
                prompt=prompt_text,
                temperature=temperature,
                model=model,
                #max_tokens=2048,
                stop=["</code>", "# SOLUTION END"],
            )
        return {'id': id, 'code': code, 'metadata': ds1000[id]['metadata']}

    print(model, idx)
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
    with open(f'data/{model}-{idx}-answers.jsonl', 'w') as f:
        for r in ds1000_responses:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-opus-20240229",
        help="which model to run")
    parser.add_argument(
        "--idx",
        type=str,
        default="1",
        help="index of sample"
    )

    ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]
    args = parser.parse_args()
    temp = 0.2
    run_claude(args.model, temp, args.idx)
  