import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

from utils import construct_dataset, TABLE_VALIDITY, DOCUMENT_VALIDITY, EXAMPLES

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset_dir",
    type=str,
    default="data/datasets/toy",
    help="Dataset directory"
)
parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    default="model_configs/valid_example_tracker.csv",
    help="Output file"
)
args = parser.parse_args()

train_dataset = construct_dataset(Path(args.dataset_dir) / "train")
test_dataset = construct_dataset(Path(args.dataset_dir) / "test")

dataset = train_dataset + test_dataset
examples = [ex["example_id"] for ex in dataset]

df = pd.DataFrame(columns=[TABLE_VALIDITY, DOCUMENT_VALIDITY])
df = df.reindex(examples)
df.index.name = EXAMPLES

df.to_csv(args.output_file)