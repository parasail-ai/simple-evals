# Overview

This repository contains a fork of [openai/simple-evals](https://github.com/openai/simple-evals) library for evaluating language models. The main addition is a `BatchChatCompletionSampler` which allow running evaluation using [OpenAI Batch API](https://platform.openai.com/docs/api-reference/batch) with OpenAI or any OpenAI compatible services, e.g. [vLLM](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_openai.md).

## Demo

```bash
# Generate OpenAI batch API requests
python -m simple-evals --models meta-llama/Meta-Llama-3-70B-Instruct,meta-llama/Meta-Llama-3-8B-Instruct --working_dir=./evals --instrument=True

# Launch OpenAI batch run, e.g using vLLM:
python -m vllm.entrypoints.openai.run_batch -i ./evals/meta-llama/Meta-Llama-3-70B-Instruct/requests.jsonl -o ./evals/meta-llama/Meta-Llama-3-70B-Instruct/responses.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct --disable-log-requests --tensor-parallel-size=4
python -m vllm.entrypoints.openai.run_batch -i ./evals/meta-llama/Meta-Llama-3-8B-Instruct/requests.jsonl -o ./evals/meta-llama/Meta-Llama-3-8B-Instruct/responses.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --disable-log-requests

# Evaluate using batch responses
python -m simple-evals --models meta-llama/Meta-Llama-3-70B-Instruct,meta-llama/Meta-Llama-3-8B-Instruct --working_dir=./evals
```

## Evals

This repository currently contains the following evals:

- MMLU: Measuring Massive Multitask Language Understanding, reference: https://arxiv.org/abs/2009.03300, https://github.com/hendrycks/test, [MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)
- MATH: Measuring Mathematical Problem Solving With the MATH Dataset, reference: https://arxiv.org/abs/2103.03874, https://github.com/hendrycks/math, [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark, reference: https://arxiv.org/abs/2311.12022, https://github.com/idavidrein/gpqa/, [MIT License](https://github.com/idavidrein/gpqa/blob/main/LICENSE)
- DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs, reference: https://arxiv.org/abs/1903.00161, https://allenai.org/data/drop, [Apache License 2.0](https://github.com/allenai/allennlp-models/blob/main/LICENSE)
- MGSM: Multilingual Grade School Math Benchmark (MGSM), Language Models are Multilingual Chain-of-Thought Reasoners, reference: https://arxiv.org/abs/2210.03057, https://github.com/google-research/url-nlp, [Creative Commons Attribution 4.0 International Public License (CC-BY)](https://github.com/google-research/url-nlp/blob/main/LICENSE)

## Benchmark Results
| model_name                           |   ('metric', 'drop') |   ('metric', 'gpqa') |   ('metric', 'math') |   ('metric', 'mgsm') |   ('metric', 'mmlu') |
|:-------------------------------------|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| meta-llama/Meta-Llama-3-70B-Instruct |             0.801795 |             0.420202 |               0.5204 |             0.828727 |               0.7956 |
| REFERENCE-RERUN                         |               |       |            |            |            |            |                     |
| Llama3 70b (report[^1])                |    0.814    |    0.413    |    0.528    |    0.826       |        0.802         |

[^1]: https://github.com/openai/simple-evals
