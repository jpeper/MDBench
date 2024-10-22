import json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    type=str,
    default="eval_output",
)
parser.add_argument(
    "-s",
    "--string",
    type=str,
    default="meta-llama",
    help="Relevant string",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default="model_configs/skip_dirs.json",
)
args = parser.parse_args()

dir_list = []

for dir in Path(args.directory).rglob("*"):
    if args.string in str(dir):
        dir_list.append(str(dir))

with open(args.file, "w") as f:
    json.dump(dir_list, f, indent=4)

