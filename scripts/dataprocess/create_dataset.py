import json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("-l", "--log_dir", type=str, required=True, help="Log directory")
parser.add_argument("-d", "--dataset_dir", type=str, default="data/datasets", help="Data directory")
parser.add_argument("-n", "--dataset_name", type=str, default="main", help="Output directory")
parser.add_argument("-o", "--overwrite", action="store_true", help="Dverwrite existing dataset")
args = parser.parse_args()

def output_example(log_file, output_dir):
    with open(log_file, "r") as f:
        log = json.load(f)

    example_dict = {}
    example_dict["question"] = log["Resultant Generated Question"]
    example_dict["ground_truth"] = log["Resultant Generated Answer"]
    example_dict["documents"] = log["Generated Document Set"]
    example_dict["table"] = log["Resultant Generated Table -- Bar-separated"]
    example_dict["characteristic_breakdown"] = json.loads(log["Example Difficulty"])
    example_dict["validity_scalar"] = log["Validity Scalar"]


    new_file_name = f"{str(log_file.parent).split('.')[0].split('raw_data/')[1].replace('/', '_')}"
    example_dict["example_id"] = new_file_name

    output_file = Path(output_dir) / f"{new_file_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(example_dict, f, indent=4)


log_dir = Path(args.log_dir)
dataset_dir = Path(args.dataset_dir) / args.dataset_name

log_files = list(log_dir.rglob("*.json"))

if args.overwrite:
    for file in dataset_dir.rglob("*"):
        if file.is_file():
            file.unlink()

    # num_files = len(log_files)
    # split_num = int(num_files * 0.6)
    split_num = 1
    train = log_files[:split_num]

    for log_file in train:
        output_dir = dataset_dir / "train"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_example(log_file, output_dir)

    test = log_files[split_num:]
else:
    test = log_files

for log_file in test:
    output_dir = dataset_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_example(log_file, output_dir)

if args.overwrite:
    print(f"Created new dataset at {dataset_dir}")
else:
    print(f"Appended {len(test)} examples to existing dataset at {dataset_dir}")