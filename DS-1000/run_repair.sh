#!/bin/bash 

#python run_ollama_repair.py --test="mypy"
#python run_ollama_repair.py --test="union"
#python run_ollama_repair.py --test="pyright"
#python run_ollama_repair.py --test="general"

#python run_openai_repair.py --test="union"
#python run_openai_repair.py --test="mypy"
#python run_openai_repair.py --test="pyright"
#python run_openai_repair.py --test="general"

python test_ds_repair.py --model="qwen3-8b" --test="mypy"
python test_ds_repair.py --model="qwen3-8b" --test="self"
python test_ds_repair.py --model="qwen3-8b" --test="pyright"

python test_ds_repair.py --model="gemma2-2b" --test="mypy"
python test_ds_repair.py --model="gemma2-2b" --test="self"
python test_ds_repair.py --model="gemma2-2b" --test="pyright"

#python test_ds_repair.py --model="o3-mini" --test="mypy"
#python test_ds_repair.py --model="o3-mini" --test="union"
#python test_ds_repair.py --model="o3-mini" --test="pyright"
#python test_ds_repair.py --model="o3-mini" --test="general"