#!/bin/bash

python run_repair_new.py --language es --model_name "gemma2-2b" --repair_type="mypy"

python run_repair_new.py --language en --model_name "gemma2-2b" --repair_type="mypy"

python run_repair_new.py --language ja --model_name "gemma2-2b" --repair_type="mypy"

python run_repair_new.py --language ru --model_name "gemma2-2b" --repair_type="mypy"
