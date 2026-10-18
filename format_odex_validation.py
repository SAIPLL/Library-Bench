import json
import os
import argparse

def from_odex_to_validator(model, static_analysis=""):
    folder_path = f"./odex/data/nl2code/{model}/"
    data_validation = []
    for file_name in os.listdir(folder_path):
        end_test = "0.json"
        if static_analysis != "":
            if static_analysis not in ['mypy', 'self', 'repair', 'pyright']:
                end_test = f'{static_analysis}-0-updated.jsonl'
            else:
                end_test = f"{static_analysis}-updated.jsonl"
        if file_name.endswith(end_test):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            print(file_path)
            # Open and read the JSON file
            if file_path.endswith('.json'):
                with open(file_path, 'r') as file:
                    data_answers = json.load(file)
            else:
                data_answers = [json.loads(item) for item in open(file_path, 'r').readlines()]
            # Open and read the test 
            lang = file_name[:2]
            for i in range(0, len(data_answers)):
                key_test = "output"
                if "outputs" in data_answers[i]:
                    key_test = "outputs"
                if file_name.endswith('updated.jsonl'):
                    data_validator = {
                        "id" : data_answers[i]['task_id'],
                        "lang" : lang,
                        "code" : data_answers[i]["predictions"],
                        "passed" : data_answers[i][key_test]["passed"],
                        "result" : data_answers[i][key_test]["result"],
                        "error_type": data_answers[i][key_test]["error_type"]
                    }
                else:
                    # Get passed and results from data_answers
                    data_validator = {
                        "id" : data_answers[i]['task_id'],
                        "lang" : lang,
                        "code" : data_answers[i]["predictions"],
                        "passed" : data_answers[i][key_test][0][1]["passed"],
                        "result" : data_answers[i][key_test][0][1]["result"],
                        "error_type": data_answers[i][key_test][0][1]["error_type"]
                    }
                data_validation.append(data_validator)
            # Save answers in format
    # Filename to write answers on
    output_name = f"./odex/data/nl2code/{model}/{model}-{static_analysis}-0-answers.jsonl"
    with open(output_name, "w") as file_write:
        for entry in data_validation:
            file_write.write(json.dumps(entry) + "\n")

def main(model, repair):
    from_odex_to_validator(model, repair)

if __name__ == "__main__":
    """ Examples """
    """ 
    python format_odex_validation.py "granite-code-latest" "grammar"
    python format_odex_validation.py "granite-code-latest" "union"
    python format_odex_validation.py "granite-code-latest"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma2-2b")
    parser.add_argument("--repair", type=str, default="0")
    args = parser.parse_args()
    main(args.model, args.repair)