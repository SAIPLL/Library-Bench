import json
import os
import pandas as pd
import argparse


def get_dataset_hf():
    splits = {'v0.1.0_hf': 'data/v0.1.0_hf-00000-of-00001.parquet', 'v0.1.1': 'data/v0.1.1-00000-of-00001.parquet', 'v0.1.2': 'data/v0.1.2-00000-of-00001.parquet', 'v0.1.3': 'data/v0.1.3-00000-of-00001.parquet', 'v0.1.4': 'data/v0.1.4-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/bigcode/bigcodebench/" + splits["v0.1.0_hf"]) 
    return df


def from_bigcode_to_validator(model, df, exp):
    #file_path = f"./bigcodebench/repair/ibm-granite/{exp}/{model}_repair_{exp}-evaluate.json"
    file_path = f"./bigcodebench/results-evaluate/{model}-{exp}-answers-sanitized_eval_results.json"
    #file_path = f"./bigcodebench/results-evaluate/{model}-repair-{exp}-0-answers-sanitized_eval_results.json"
    print(file_path)
    data_validation = []
    with open(file_path, 'r') as file:
        data_answers = json.load(file)
    for i, task_info in df.iterrows():
        # Get passed and results from data_answers
        task_info_id = task_info['task_id']
        if task_info_id not in data_answers['eval']:
            continue
        results_id = data_answers['eval'][task_info_id][0]
        result_details = results_id["details"]
        if result_details:
            error_msg = result_details[next(iter(result_details))]
            if len(error_msg) == 0:
                error_msg = ''
        else:
            error_msg = ''
        data_validator = {
            "id" : task_info_id,
            "code" : results_id['solution'],
            "passed" : results_id["status"],
            "result" : error_msg,
            "metadata" : task_info.to_json()
        }
        data_validation.append(data_validator)
            # Save answers in format
    # Filename to write answers on
    #output_name = f"./bigcodebench/repair/ibm-granite/{exp}/{model}-answers.jsonl"
    #output_name = f"./bigcodebench/results/{model}-answers-validation.json"
    output_name = f"./bigcodebench/results-evaluate/{model}_{exp}-bigcode-answers.jsonl"
    print(output_name)
    with open(output_name, "w") as file_write:
        for entry in data_validation:
            file_write.write(json.dumps(entry) + "\n")

def main(model, repair):
    # Get dataset from huggingface
    df = get_dataset_hf()
    from_bigcode_to_validator(model, df, repair)

if __name__ == "__main__":
    """ Examples """
    """ 
    python format_bigcodebench_validation.py "granite-code-latest" "grammar"
    python format_bigcodebench_validation.py "granite-code-latest" "union"
    python format_bigcodebench_validation.py "granite-code-latest"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma2-2b")
    parser.add_argument("--repair", type=str, default="0")
    args = parser.parse_args()
    main(args.model, args.repair)