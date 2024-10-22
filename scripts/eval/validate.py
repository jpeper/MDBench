import pandas as pd
from argparse import ArgumentParser
from json import dump
from pathlib import Path
from statistics import mean
from utils import DATASETS_PATH, TABLE_VALIDITY, EXAMPLES, construct_dataset

parser = ArgumentParser()
parser.add_argument("-csv", "--valid_example_tracker_path", type=str, default="eval_output/valid_example_tracker.csv", help="Valid example tracker CSV path")
parser.add_argument("-d", "--dataset", type=str, default="main", help="Dataset name")
parser.add_argument("-o", "--output_file", type=str, default="eval_output/validity_eval.json", help="Output file")
args = parser.parse_args()

dataset_path = Path(DATASETS_PATH) / args.dataset

train_dataset = construct_dataset(dataset_path / "train")
test_dataset = construct_dataset(dataset_path / "test")

dataset = train_dataset + test_dataset
valid_df = pd.read_csv(args.valid_example_tracker_path)
valid_df.set_index(EXAMPLES, inplace=True)
matches = []
match_dict = {"proportion_matching": 0, "valid_proportion_matching": 0, "proportion_valid": 0, "proportion_valid_llm": 0}

human_table_validities = []
llm_table_validities = []

for ex in dataset:
    id = ex["example_id"]
    del ex["example_id"]

    if pd.isna(valid_df.loc[id, TABLE_VALIDITY]):
        continue

    human_table_validity = int(valid_df.loc[id, TABLE_VALIDITY])
    llm_table_validity = ex["validity_scalar"] == 4 or ex["validity_scalar"] == 5

    matches.append(int(human_table_validity == llm_table_validity))
    human_table_validities.append(human_table_validity)
    llm_table_validities.append(int(llm_table_validity))

    match_dict[id] = ex
    match_dict[id]["human_validity"] = human_table_validity
    match_dict[id]["correct_validity"] = int(human_table_validity == llm_table_validity)

match_dict["proportion_matching"] = mean(matches)
match_dict["proportion_valid"] = mean(human_table_validities)
match_dict["proportion_valid_llm"] = mean(llm_table_validities)

valid_intersection = 0
for human_val, llm_val in zip(human_table_validities, llm_table_validities):
    if human_val and llm_val:
        valid_intersection += 1

match_dict["valid_proportion_matching"] = valid_intersection / (match_dict["proportion_valid_llm"] * len(llm_table_validities))

with open(args.output_file, "w") as f:
    dump(match_dict, f, indent=4)



