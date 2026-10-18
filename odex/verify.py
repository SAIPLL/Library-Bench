"""Verify the correctness of model predictions. """

import os
import re
import random
import evaluate
from typing import Dict, List 
from processor import CodeProcessor
from prompt import add_indent, get_entry_point 

code_eval_metric = evaluate.load("code_eval")
# os.environ["HF_ALLOW_CODE_EVAL"] = "0"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_valid_solutions(
    predictions: List[str], 
    deduplicate: bool = False, 
) -> List[str]: 
    processor = CodeProcessor()
    solutions = [] 
    for pred in predictions: 
        valid_sol = processor.code_extract(pred)
        solutions.append(valid_sol)
    
    if deduplicate: 
        solutions = list(set(solutions))
    return solutions

def wrap_check(
    sample: Dict, 
    solution_list: List[str], 
    k: List[int], 
    num_workers: int = 1, 
    max_num_tests: int = 1, 
    verbose: bool = True, 
    exclude_suffix: bool = True, 
    function_name: str = "id", 
): 
    if exclude_suffix: 
        wrapped_solution_list = [
            f"{solution}".replace('\t', ' '*4) 
            for solution in solution_list
        ]
    else: 
        wrapped_solution_list = [
            f"{solution}".replace('\t', ' '*4) 
            for solution in solution_list
        ]

    max_num_tests = min(len(sample["test"]), max_num_tests)
    test_case = random.sample(sample["test"], max_num_tests)
    entry_point = get_entry_point(sample, function_name)

    # Wrap check function to capture structured error info
    check_function = '\n'.join([
        "import traceback, sys",
        sample['test_start'],
        ''.join(test_case),
        '',
        f"""
def _safe_check():
    try:
        check({entry_point})
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        tb = "".join(traceback.format_tb(e.__traceback__))
        raise type(e)(
            f"[ErrorType]: {{error_type}}\\n"
            f"[Message]:   {{error_msg}}\\n"
            f"[Traceback]:\\n{{tb}}"
        )

_safe_check()
""",
    ])

    scores, outputs = code_eval_metric.compute(
        predictions=[wrapped_solution_list], 
        references=[check_function], 
        k=k, 
        num_workers=num_workers, 
        timeout=20
    )

    # Parse error type out of each result string
    parsed_outputs = []
    for candidate_idx, result in outputs[0]:
        passed = result["passed"]
        result_msg = result["result"]
        
        error_type = None
        if not passed:
            # 1. Try structured [ErrorType] tag injected by _safe_check()
            for line in result_msg.splitlines():
                if line.startswith("[ErrorType]:"):
                    error_type = line.replace("[ErrorType]:", "").strip()
                    break

            # 2. Fallback: parse "failed: SomeError(...)" or "failed: SomeException(...)"
            if error_type is None:
                match = re.search(r'\b(\w*Error|\w*Exception)\b', result_msg)
                if match:
                    error_type = match.group(1)

            # 3. Fallback: infer from known message patterns
            if error_type is None:
                pattern_map = [
                    (r"has no attribute",          "AttributeError"),
                    (r"is not callable",           "TypeError"),
                    (r"unsupported operand type",  "TypeError"),
                    (r"No module named",           "ImportError"),
                    (r"cannot import name",        "ImportError"),
                    (r"name '.*?' is not defined", "NameError"),
                    (r"index out of range",        "IndexError"),
                    (r"key error",                 "KeyError"),
                    (r"division by zero",          "ZeroDivisionError"),
                    (r"timed out",                 "TimeoutError"),
                    (r"syntax",                    "SyntaxError"),
                    (r"object cannot be interpreted  as", "TypeError"),
                    (r"must be.*?, not",            "TypeError"),
                    (r"unexpected indent",           "IndentationError"),
                    (r"object is not subscriptable", "TypeError"),
                    (r"not supported between", "TypeError"),
                    (r"operand", "TypeError"),
                    (r"required positional argument", "TypeError"),
                    (r"not enough values to unpack", "TypeError"),
                    (r"object of type", "TypeError") #object of type 'NoneType' has no len()
                ]
                for pattern, inferred_type in pattern_map:
                    if re.search(pattern, result_msg, re.IGNORECASE):
                        error_type = inferred_type
                        break

            # 4. Last resort
            if error_type is None:
                error_type = "UnknownError"

        parsed_outputs.append({
            "candidate_idx": candidate_idx,
            "passed":        passed,
            "error_type":    error_type,   # e.g. "ImportError", "TypeError", None if passed
            "result":        result_msg,
        })

    if verbose: 
        print(f"[predic] {wrapped_solution_list}")
        print(f"[fcheck] {check_function}")
        print(f"[scores] {scores}")
        for o in parsed_outputs:
            status = "✓" if o["passed"] else f"✗ {o['error_type']}"
            print(f"[output] candidate={o['candidate_idx']} | {status} | {o['result']}")

    return scores, parsed_outputs[0]


def wrap_check_test(
    prompt: str, suffix: str, 
    solution_list: List[str], 
    test_start: str, 
    test_case: str, 
    entry_point: str, 
    k: List[int] = [1], 
    num_workers: int = 1, 
    add_indent_test: bool = True, 
    verbose: bool = False, 
): 
    wrapped_solution_list = [
        f"{prompt}{solution}{suffix}".replace('\t', ' '*4) 
        for solution in solution_list
    ]
    
    if add_indent_test == True: 
        test_case = add_indent(test_case)
    check_function = '\n'.join([
        test_start, 
        test_case, 
        '', 
        f"check({entry_point})", 
    ])
    if verbose:
        print(f"[solution list] \n{wrapped_solution_list}")
        print(f"[check function] \n{check_function}")
    scores, outputs = code_eval_metric.compute(
        predictions=[wrapped_solution_list], 
        references=[check_function], 
        k=k, 
        num_workers=num_workers, 
    )
    if verbose: 
        print(f"[scores] {scores}")
        print(f"[output] {outputs[0]}")
        print('-'*25)
    return scores, outputs[0]