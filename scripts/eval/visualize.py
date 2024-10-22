import json
import pandas as pd
from pathlib import Path
from copy import deepcopy
from statistics import mean

from utils import construct_dataset, visualize_args, LEVELS, LEVELS_LIST, DATASETS_PATH


class Visualize:
    def __init__(
        self,
        eval_output_dir,
        skip_dirs=[],
        include_dirs=[],
        output_path="eval_visualize",
        dataset="main",
        validity="all"
    ):
        self.eval_output_dir = Path(eval_output_dir) / Path(validity) / dataset
        self.skip_dirs = skip_dirs
        self.include_dirs = include_dirs
        self.output_dir = Path(output_path) / Path(validity) / dataset
        self.output_files = []
        self.dataset_dir = Path(DATASETS_PATH) / dataset
        self.cb = "characteristic_breakdown"
        self.doc_reasoning = "doc_reasoning"
        self.table_reasoning = "table_reasoning"


    def reasoning_results(self):
        self.eval_output_dir = self.eval_output_dir / "documents"
        self.__set_output_files()
        df, EM_df = self.__reasoning_results()
        self.__to_csv("doc_reasoning_results", df)
        self.__to_csv("doc_reasoning_results_EM", EM_df)

        self.eval_output_dir =  self.eval_output_dir.parent / "table"
        self.__set_output_files()
        df, EM_df = self.__reasoning_results()
        self.__to_csv("table_reasoning_results", df)
        self.__to_csv("table_reasoning_results_EM", EM_df)

    def characteristic_breakdown(self, prompt_type="zero-shot-CoT", characteristic="multi-hop"):
        if "table" in str(self.eval_output_dir):
            self.eval_output_dir = self.eval_output_dir.parent   

        self.eval_output_dir = self.eval_output_dir / "documents"
        self.__set_output_files()
        df = self.__characteristic_breakdown_results(prompt_type, characteristic)
        self.__to_csv(f"{self.cb}/{self.doc_reasoning}|{characteristic}|{prompt_type}", df)

        self.eval_output_dir = self.eval_output_dir.parent / "table"
        self.__set_output_files()
        df = self.__characteristic_breakdown_results(prompt_type, characteristic)
        self.__to_csv(f"{self.cb}/{self.table_reasoning}|{characteristic}|{prompt_type}", df)

    def overall_characteristic_breakdown(self, characteristic="multi-hop"):
        assert Path(self.output_dir / self.cb).is_dir() and \
            (list(Path(self.output_dir / self.cb).iterdir()) != [])

        df = self.__overall_characteristic_breakdown_results(self.doc_reasoning, characteristic)   
        self.__to_csv(f"{self.cb}/{self.doc_reasoning}|{characteristic}|overall", df)     
        df = self.__overall_characteristic_breakdown_results(self.table_reasoning, characteristic)
        self.__to_csv(f"{self.cb}/{self.table_reasoning}|{characteristic}|overall", df)

    def dataset_characteristic_breakdown(self):
        train_dataset = construct_dataset(Path(self.dataset_dir) / "train")
        test_dataset = construct_dataset(Path(self.dataset_dir) / "test")

        dataset = train_dataset + test_dataset
        dataset_characteristics = {}

        dataset = json.load(open('characteristics_dump.json'))
        for datum in dataset.values():
            for key, value in datum.items():
                if key not in dataset_characteristics:
                    dataset_characteristics[key] = deepcopy(LEVELS)
                dataset_characteristics[key][value] += 1




        df = pd.DataFrame(dataset_characteristics)
        self.__to_csv(f"{self.cb}/dataset_characteristic_breakdown", df)

    def __set_output_files(self):
        if self.include_dirs:
            self.output_files = [
                f
                for f in self.eval_output_dir.rglob("*.json")
                if str(f) in self.include_dirs
            ]
        else:
            self.output_files = [
                f
                for f in self.eval_output_dir.rglob("*.json")
                if str(f) not in self.skip_dirs
            ]

    def __to_csv(self, output_filename, df):
        print(df)
        output_path = Path(self.output_dir) / f"{output_filename}.csv"
        print(f"Outputting to {str(output_path)}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)

    def __reasoning_results(self):
        data = []
        data_EM = []

        for f in self.output_files:
            model = f.parts[-2]
            prompt_type = f.name.split("_")[0]

            with open(f, "r") as f:
                eval_scores = json.load(f)

            data.append(
                {
                    "model": model,
                    "prompt_type": prompt_type,
                    "avg_accuracy": eval_scores["accuracy"]["mean"],
                }
            )

            try:
                evaluations = eval_scores["evaluations"].values()
            except:
                evaluations = eval_scores["evaluations"]
            num_EM = 0
            for eval in evaluations:
                if eval["accuracy"] == 10 or eval["accuracy"] == 9:
                    num_EM += 1

            data_EM.append(
                {
                    "model": model,
                    "prompt_type": prompt_type,
                    "EM": (num_EM / len(evaluations))
                }
            )

        df = pd.DataFrame(data)
        df = df.pivot(index="model", columns="prompt_type", values="avg_accuracy")
        df['Overall'] = df.mean(axis=1)
        df = df * 10

        # Specify the order of the columns
        df = df[['zero-shot', 'zero-shot-CoT', 'few-shot', 'few-shot-CoT', 'Overall']]

        EM_df = pd.DataFrame(data_EM)
        EM_df = EM_df.pivot(index="model", columns="prompt_type", values="EM")
        EM_df['Overall'] = EM_df.mean(axis=1)
        EM_df = EM_df * 100

        # Specify the order of the columns
        EM_df = EM_df[['zero-shot', 'zero-shot-CoT', 'few-shot', 'few-shot-CoT', 'Overall']]

        return df, EM_df

    def __characteristic_breakdown_results(self, prompt_type, characteristic):
        prompt_output_files = [f for f in self.output_files if prompt_type == f.name.split("_")[0]]

        eval_characteristics = {}

        for f in prompt_output_files:
            model_name = f.parts[-2]
            if model_name not in eval_characteristics:
                eval_characteristics[model_name] = deepcopy(LEVELS_LIST)

            with open(f, "r") as file:
                eval_scores = json.load(file)


            sourced_breakdown = json.load(open("characteristics_dump.json"))
            for ex_id, eval in eval_scores["evaluations"].items():
                characteristic_breakdown = sourced_breakdown[ex_id]
                if isinstance(eval["accuracy"], int):
                    eval_characteristics[model_name][characteristic_breakdown[characteristic]].append(eval["accuracy"])
                else:
                    eval_characteristics[model_name][characteristic_breakdown[characteristic]].append(0)

        def list_average(lst):
            if len(lst) == 0:
                return None  # Return 0 for empty lists
            return mean(lst)
            
        df = pd.DataFrame(eval_characteristics)
        # Apply the list_average function only to columns with lists
        for col in df.columns:
            df[col] = df[col].apply(list_average)

        df.index.name = "Difficulty Level"

        df = df * 10
        
        return df

    def __overall_characteristic_breakdown_results(self, reasoning_type, characteristic):
        dfs = []
        for csv_file in Path(self.output_dir / self.cb).rglob(f"**/{reasoning_type}*{characteristic}*.csv"):
            print(str(csv_file))
            if "overall" in str(csv_file):
                continue
            df = pd.read_csv(csv_file, index_col=0)
            dfs.append(df)

        assert len(dfs) == 4
        
        overall_df = sum(dfs) / len(dfs)
        return overall_df


def main(args):
    skip_dirs = []
    include_dirs = []

    if args.skip_dir_file:
        with open(args.skip_dir_file, "r") as f:
            skip_dirs = json.load(f)

    if args.include_dir_file:
        with open(args.include_dir_file, "r") as f:
            include_dirs = json.load(f)

    vis = Visualize(args.eval_output_dir, skip_dirs, include_dirs, args.output_path, args.dataset, args.validity)

    if args.reasoning_results:
        vis.reasoning_results()
    if args.characteristic_breakdown:
        vis.characteristic_breakdown(args.prompt_type, args.characteristic)
    if args.overall_characteristic_breakdown:
        vis.overall_characteristic_breakdown(args.characteristic)
    if args.dataset_characteristic_breakdown:
        vis.dataset_characteristic_breakdown()


if __name__ == "__main__":
    args = visualize_args()
    main(args)
