import json
import os

import pandas as pd
from absl import app, flags

from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .math_eval import MathEval
from .sampler.batch_chat_completion_sampler import BatchChatCompletionSampler

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "models", "meta-llama/Meta-Llama-3-8B-Instruct", "List of model names to evaluate."
)

flags.DEFINE_boolean(
    "instrument",
    False,
    "If True, only output 'requests.jsonl' and skip the sampling and evalution.",
)

flags.DEFINE_integer("max_tokens", 2048, "Maximum number of generated tokens.")

flags.DEFINE_float("temperature", 0.5, "Sampling temperature.")

flags.DEFINE_string("working_dir", "/tmp", "Working dir for eval results.")


def main(argv: list[str]):
    evals = {
        "mmlu": MMLUEval(num_examples=2500),
        "gpqa": GPQAEval(n_repeats=10),
        "mgsm": MGSMEval(num_examples_per_lang=250),
        "drop": DropEval(num_examples=2000, train_samples_per_prompt=3),
        "math": MathEval(num_examples=2500),
    }

    mergekey2resultpath = {}
    for model in FLAGS.models:
        working_dir = f"{FLAGS.working_dir}/{model}"
        os.makedirs(working_dir, exist_ok=True)

        sampler = BatchChatCompletionSampler(
            model=model,
            working_dir=working_dir,
            temperature=FLAGS.temperature,
            max_tokens=FLAGS.max_tokens,
            instrument=FLAGS.instrument,
        )
        results = {name: eval(sampler) for name, eval in evals.items()}
        if FLAGS.instrument:
            print(f"Requests written to {working_dir}/requests.jsonl")
            continue

        for eval_name, result in results.items():
            eval_model_name = f"{eval_name}/{model}"
            output_dir = f"{working_dir}/{eval_name}/"
            os.makedirs(output_dir, exist_ok=True)

            report_filename = output_dir + "report.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))

            metrics = result.metrics | {"score": result.score}
            result_filename = output_dir + "result.json"
            print(f"Writing results to {result_filename}")
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))

            mergekey2resultpath[f"{eval_model_name}"] = result_filename

    if FLAGS.instrument:
        print("Instrument run finished")
        return

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue

        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("/")]
        model_name = eval_model_name[eval_model_name.find("/") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )

    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())


if __name__ == "__main__":
    app.run(main)
