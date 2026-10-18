# Library Benchmarks

### [DS-1000](https://github.com/xlang-ai/DS-1000#385bba98025a18b7bbcfb5c63fb587751913c024)

I have edited ```DS-1000/execution.py``` and ```DS-1000/test_ds1000.py``` to provide more information and format about the error message.
I have included a call to the Anthropic API, OpenAI API, and Gemini API through this script ```DS-1000/run_claude.py```.
I have included a call to the Ollama API (running locally) through this script ```DS-1000/run_ollama.py```.

```
cd DS-1000
python run_ollama.py
```

To run ollama with the grammar for a specific library

```
python run_ollama.py --grammar "True" --library "Numpy"
```
For any library:
```
python run_ollama.py --grammar "True" --library ""
```

### [Odex](https://github.com/zorazrw/odex#21308ea6d474a078d5a85c61c2b5ca228d1fd167)

I have edited Odex ```odex/nl2code_concurrent_claude.py``` to work with OpenAI's chat API, Anthropic Messages, and Gemini API. As in pass@1 the default `temperature=0.2` and `n=1`. I have skipped the prompts in the benchmark that don't belong to the `"open"` domain.

```
cd odex
python nl2code_concurrent_claude.py --language es --model_name "claude-3-opus-20240229"
python nl2code_concurrent_claude.py --language es --model_name "gpt-4-0613"
python nl2code_concurrent_claude.py --language es --model_name "gpt-3.5-turbo-0125"
python nl2code_concurrent_claude.py --language es --model_name "granite-code:latest"
python nl2code_concurrent_claude.py --language es --model_name "granite-code:latest" --grammar True
python nl2code_concurrent_claude.py --language es --model_name "gemma2:2b" --output_path "data/nl2code/gemma2-2b/es_pred_0-0.jsonl"
```

Run odex re-evaluation.

```
pip install -r ./library/requirements.txt 
python ./library/imports.py

python eval_passk.py --language es --prediction_path data/nl2code/claude-3-opus-20240229/es_pred_0-0.json --library_usage "open"
```

To format the results in odex to work with the validator script.
```
python format_odex_validation.py
```

### [BigCodeBench](https://github.com/bigcode-project/bigcodebench#47eff25a9933ba89eee755bcf2696c4f17160b54)

I followed this [setup](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md). 

```
cd bigcodebench
bigcodebench.evaluate --execution local --split instruct --subset full --samples sample/claude-3-opus-20240229--bigcodebench-instruct--anthropic-0-1-sanitized-calibrated.jsonl
bigcodebench.evaluate --execution local --split instruct --subset full --samples sample/gpt-4-0613--bigcodebench-instruct--openai-0-1-sanitized-calibrated.jsonl
bigcodebench.evaluate --execution local --split instruct --subset full --samples sample/gpt-3.5-turbo-0125--bigcodebench-instruct--openai-0-1-sanitized-calibrated.jsonl
bigcodebench.evaluate --execution local --split instruct --subset full --samples sample/ibm-granite--granite-3.0-2b-instruct--bigcodebench-instruct--vllm-0-1-sanitized_calibrated.jsonl
```

To format the results in BigCodeBench to work with the validator script.
```
python format_bigcodebench_validation.py
```

To generate more samples use this script:
```
python run_claude.py --model='gemini-3.5-flash-lite'
```