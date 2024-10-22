import pickle
import json
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
try:
    from prompts import PROMPTS
except:
    from llm.prompts import PROMPTS

ENV_PATH = "model_configs/.env"
CONTEXT_KEY = "Context:"
INPUT_KEY = "Q:"
OUTPUT_KEY = "A:"

API_INSTRUCTION_IDX = 0


def get_args():
    parser = ArgumentParser()

    # basic model params
    parser.add_argument("--model", type=str, required=True, help="LLM model")
    parser.add_argument("--name", type=str, required=False, help="Model name")
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="Tokenizer name"
    )
    parser.add_argument("--task", type=str, default=None, help="Task name")

    # additional model params
    parser.add_argument(
        "--remote", action="store_true", help="Whether to trust remote code"
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max length")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens"
    )

    # task params
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Dataset folder"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file")

    # prompt variations
    parser.add_argument(
        "-k", "--k_examples", type=int, default=1, help="Number of examples"
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to choose from.",
    )
    parser.add_argument(
        "-x", "--x_shot_prompt", type=str, choices=PROMPTS.keys(), help="Prompt type"
    )
    parser.add_argument(
        "-r",
        "--reasoning_type",
        type=str,
        choices=["documents", "table"],
        help="Reasoning type",
    )
    parser.add_argument(
        "-api",
        "--is_api_model",
        action="store_true",
        help="Is model accessed via API"
    )
    parser.add_argument(
        "-a",
        "--num_to_append",
        type=int,
        default=0,
        help="Number of examples to append",
    )

    args = parser.parse_args()
    return args


def construct_dataset(dataset_path):
    dataset = []

    for file in sorted(Path(dataset_path).iterdir()):
        with open(file, "r") as f:
            dataset.append(json.load(f))

    return dataset


def flatten_output(output):
    if type(output[0]) is list:
        return list(chain(*output))
    return output


def dump_output(output_file, outputs, append=False):
    # if Path(output_file).exists() and not append:
    #     back_int = 0
    #     while Path(f"{output_file}.{back_int}.bak").exists():
    #         back_int += 1
    #     Path(output_file).rename(f"{output_file}.{back_int}.bak")
    #     assert not Path(output_file).exists()

    print(output_file)

    if "/" in output_file:
        Path(output_file[: output_file.rfind("/")]).mkdir(parents=True, exist_ok=True)

    if append:
        with open(output_file, "r") as f:
            old_outputs = json.load(f)
        outputs = old_outputs + outputs

    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)
