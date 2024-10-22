from argparse import ArgumentParser
from nltk import word_tokenize
from pathlib import Path
from pipeline import Pipeline
from utils import construct_dataset
from prompts import PROMPTS

parser = ArgumentParser()
parser.add_argument(
    "-d", "--dataset_path", type=str, default="data/datasets/toy", help="Dataset file"
)
parser.add_argument("-x", "--x_shot_prompt", type=str, choices=PROMPTS.keys(),default="few-shot", help="Prompt type")

args = parser.parse_args()

class Args:
    def __init__(self, dataset_path, x_shot_prompt):
        self.model = "llama-2"
        self.name = "llama-2"
        self.tokenizer_name = "llama-2"
        self.task = "text-generation"

        self.max_length = 512
        self.max_new_tokens = 100
        self.remote = False

        self.dataset_path = dataset_path
        self.x_shot_prompt = x_shot_prompt
        self.reasoning_type = "table"
        self.num_to_append = 0
        self.is_api_model = False

        self.k_examples = 1
        self.limit = 1

pipeline = Pipeline(Args(args.dataset_path, args.x_shot_prompt))
prompts, _ = pipeline.get_prompts()
total_tokens = 0
max_tokens = 0

for prompt in prompts:
    total_tokens += len(word_tokenize(prompt))
    max_tokens = max(max_tokens, len(word_tokenize(prompt)))

print(prompts[0])

print(f"Total input tokens: {total_tokens}")
print(f"Avg tokens per prompt: {total_tokens/len(prompts)}")
print(f"Max tokens per prompt: {max_tokens}")

dataset = construct_dataset(Path(args.dataset_path) / "test")

total_out_tokens = 0
max_out_tokens = 0
for data in dataset:
    answer = data["ground_truth"]
    total_out_tokens += len(word_tokenize(answer))
    max_out_tokens = max(max_out_tokens, len(word_tokenize(answer)))

print(f"Total output tokens: {total_out_tokens}")
print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")
print(f"Max out tokens per prompt: {max_out_tokens}")
