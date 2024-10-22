from argparse import ArgumentParser
from pathlib import Path
from sys import path
path.insert(1, "./scripts/")

from llm.utils import construct_dataset, ENV_PATH
from llm.prompts import PROMPTS

from dataprocess.utils import TABLE_VALIDITY, EXAMPLES

# LEVELS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
# LEVELS_LIST = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

LEVELS = {0: 0, 1: 0}
LEVELS_LIST = {0: [], 1: []}

DATASETS_PATH = "data/datasets"

ALL = "all"
NO_INV = "no_invalid"
ONLY_VAL = "only_valid"

VALIDITY = [ALL, NO_INV, ONLY_VAL]
VALIDITY_CSV_PATH = Path("eval_output/valid_example_tracker.csv")

ANSWER_KEY = "ANSWER: "

def evaluate_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_output_file", type=str, required=True, help="Model output file"
    )
    parser.add_argument(
        "--eval_output_file", type=str, required=True, help="Eval output file"
    )
    parser.add_argument(
        "--num_to_append",
        type=int,
        default=0,
        help="Number of examples to append to the model output file",
    )
    parser.add_argument(
        "--validity",
        type=str,
        choices=VALIDITY,
        default="all",
        help="Validity setting"
    )
    parser.add_argument(
        "--x_shot_prompt", type=str, required=True, help="Prompt type"
    )
    args = parser.parse_args()
    return args


def visualize_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--reasoning_results",
        action="store_true",
        help="Visualize reasoning results",
    )
    parser.add_argument(
        "-cb",
        "--characteristic_breakdown",
        action="store_true",
        help="Visualize characteristics breakdown",
    )
    parser.add_argument(
        "-ocb",
        "--overall_characteristic_breakdown",
        action="store_true",
        help="Visualize overall characteristics breakdown"
    )
    parser.add_argument(
        "-dcb",
        "--dataset_characteristic_breakdown",
        action="store_true",
        help="Visualize dataset characteristics breakdown",
    )
    parser.add_argument(
        "-dd",
        "--dataset",
        type=str,
        default="main",
        help="Dataset directory",
    )
    parser.add_argument(
        "-e",
        "--eval_output_dir",
        type=str,
        default="eval_output",
        help="Eval output directory",
    )
    parser.add_argument(
        "-sdf",
        "--skip_dir_file",
        type=str,
        default="model_configs/skip_dirs.json",
        help="File with directories to skip",
    )
    parser.add_argument(
        "-idf",
        "--include_dir_file",
        type=str,
        default=None,
        help="File with directories to include",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="eval_visualize", help="Output file"
    )
    parser.add_argument(
        "-p", "--prompt_type", type=str, choices=PROMPTS.keys(), default="zero-shot-CoT", help="Prompt type"
    )
    parser.add_argument(
        "-c",
        "--characteristic",
        type=str,
        choices=[
            "multi-hop",
            "temporal",
            "numeric",
            "information_aggregation",
            "soft_reasoning",
        ],
        default="multi-hop",
        help="Characteristic",
    )
    parser.add_argument(
        "-v",
        "--validity",
        type=str,
        choices=VALIDITY,
        default="all",
        help="Validity setting"
    )
    args = parser.parse_args()
    return args
