import torch
import time
import json
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from nltk import word_tokenize
from transformers.generation import GenerationConfig
from pathlib import Path
from random import sample, seed
from api_pipeline import is_concurrent, query_api
from prompts import PROMPTS, CoT
from utils import CONTEXT_KEY, INPUT_KEY, OUTPUT_KEY, construct_dataset, flatten_output

device = 0 if torch.cuda.is_available() else -1
seed(0)

class Pipeline:
    def __init__(self, args):
        self.model = args.model
        self.name = args.name
        self.tokenizer_name = args.tokenizer_name
        self.task = args.task

        self.max_length = args.max_length
        self.max_new_tokens = args.max_new_tokens
        self.remote = args.remote

        self.dataset_path = Path(args.dataset_path)
        self.x_shot_prompt = args.x_shot_prompt
        self.reasoning_type = args.reasoning_type

        self.k_examples = args.k_examples
        self.limit = args.limit
        self.num_to_append = args.num_to_append

        self.is_api_model = args.is_api_model


    def get_model_output(self):
        prompts, prompts_dict = self.get_prompts()
        if self.is_api_model:
            output = query_api(
                prompts,
                self.model,
                self.name,
                max_tokens=self.max_new_tokens,
                x_shot_prompt=self.x_shot_prompt,
                serial=not is_concurrent(self.model),
                parallel=is_concurrent(self.model)
            )
        else:
            output = self.run_pipeline(prompts)

        return output, prompts_dict

    def get_prompts(self):
        test_dataset = construct_dataset(self.dataset_path / "test")

        if self.num_to_append:
            test_dataset = test_dataset[-self.num_to_append:]

        print("Processing dataset...")

        if "zero" in self.x_shot_prompt:
            instruction = PROMPTS["zero-shot"]
        elif "few" in self.x_shot_prompt:
            instruction = PROMPTS["few-shot"]

        prompts = []

        if self.is_api_model: 
            prompts.append(instruction)

        train_dataset = self.__get_train_dataset() 
        prompts_dict = []

        for _, data in enumerate(tqdm(test_dataset, desc="Processing Dataset", unit="item")):
            context = data[self.reasoning_type]
            question = data["question"]
            
            del data["table"]
            del data["documents"]

            data["context"] = context

            prompts_dict.append(data)

            examples = []
            if "few" in self.x_shot_prompt:
                examples = self.__get_examples(train_dataset)

            if self.is_api_model:
                prompt = {
                    "context": context,
                    "question": question,
                    "examples": examples,
                }
            else:
                prompt = self.__get_prompt(instruction, context, question, examples)

            prompts.append(prompt)
        return prompts, prompts_dict

    def run_pipeline(self, prompts):
        # Initialize the pipeline with the specified model, and set the device
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, model_max_length=self.max_length
        )
        # model = AutoModelForCausalLM.from_pretrained(args.model)
        gen_config = GenerationConfig.from_pretrained(self.model)
        gen_config.max_new_tokens = self.max_new_tokens
        gen_config.max_length = self.max_length
        # pre_config = PretrainedConfig.from_pretrained(args.model)

        print("Initializing pipeline...")

        model_pipe = pipeline(
            self.task,
            model=self.model,
            tokenizer=tokenizer,
            device_map="auto",
            trust_remote_code=self.remote,
            # config=pre_config,
        )

        print("Running pipeline...")

        start_time = time.time()
        model_pipe(prompts[:1], generation_config=gen_config)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"First prompt: {elapsed_time} seconds")

        output = model_pipe(prompts, generation_config=gen_config)

        flat_output = flatten_output(output)
        total_out_tokens = 0
        cleaned_outputs = []
        for out in flat_output:
            clean_output = out["generated_text"].split(OUTPUT_KEY)[-1].strip()
            total_out_tokens += len(word_tokenize(clean_output))
            cleaned_outputs.append(clean_output)
        print(f"Total output tokens: {total_out_tokens}")
        print(f"Avg out tokens per prompt: {total_out_tokens/len(prompts)}")

        return cleaned_outputs

    ### HELPERS ###
    def __get_prompt(self, instruction, context, question, examples=[]):
        example_str = ""
        for ex in examples:
            example_str += f"{CONTEXT_KEY} {ex['context']}\n{INPUT_KEY} {ex['question']}\n{OUTPUT_KEY} {ex['ground_truth']}\n"

        if "CoT" in self.x_shot_prompt:
            formatted_prompt = f"{instruction}\n{example_str}{CONTEXT_KEY} {context}\n{INPUT_KEY} {question}\n{CoT}\n{OUTPUT_KEY}"
        else:
            formatted_prompt = f"{instruction}\n{example_str}{CONTEXT_KEY} {context}\n{INPUT_KEY} {question}\n{OUTPUT_KEY}"

        return formatted_prompt

    def __get_train_dataset(self):
        train_dataset = construct_dataset(self.dataset_path / "train")

        if self.limit:
            limit_indices = sample(range(len(train_dataset)), k=self.limit)
            train_dataset = [train_dataset[i] for i in limit_indices]
            assert len(train_dataset) == self.limit

        return train_dataset

    def __get_examples(self, train_dataset):
        if self.limit:
            assert len(train_dataset) == self.limit

        examples = []
        
        indices = sample(range(len(train_dataset)), k=self.k_examples)

        for idx in indices:
            context = train_dataset[idx][self.reasoning_type]
            question = train_dataset[idx]["question"]
            answer = train_dataset[idx]["ground_truth"]
            
            examples.append({
                "context": context,
                "question": question,
                "ground_truth": answer
            })
        
        return examples
