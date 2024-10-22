import time
import asyncio
import json
import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

from statistics import mean, median, stdev
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
from llm_infer import llm_infer

from utils import ENV_PATH, EXAMPLES, NO_INV, ONLY_VAL, VALIDITY_CSV_PATH, TABLE_VALIDITY, ANSWER_KEY, evaluate_args

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def create_eval_query(model_response, ground_truth, query):
    norm_val_query = f"Does the predicted answer align with the correct answer as it relates to the original question? Provide an answer (max length of 3 sentences).\n Question{query} \nPredicted: {model_response}\nCorrect: {ground_truth}"
    scalar_val_query = f"On a scale of 0-10 how well does the predicted answer align with the correct answer as it relates to the original question? RETURN ONLY AN INTEGER.\nQuestion{query}\nPredicted: {model_response}\nCorrect: {ground_truth}"

    return norm_val_query, scalar_val_query

def create_compression_query(model_response, question):
    return f"Take this potentially verbose answer and concisely return just the final answer to the provided question -- usually in a couple of sentences.\nQuestion: {question}\nAnswer: {model_response}"

def send_eval_queries(examples, x_shot_prompt="zero-shot"):
    norm_val_queries = []
    scalar_val_queries = []

    compression_queries = []
    for ex in examples.values():
        
        if "CoT" in x_shot_prompt and ANSWER_KEY in ex["model_response"]:
            ex["model_response"].replace("*", "")

            ex["rationale"] = ex["model_response"].split(ANSWER_KEY)[0]
            ex["model_response"] = ex["model_response"].split(ANSWER_KEY)[1]
        compression_queries.append(create_compression_query(ex["model_response"], ex["question"]))

    compression_results = llm_infer(compression_queries)


    for idx, ex in enumerate(examples.values()):
        ex["rationale"] = None
        ex["model_response"] = compression_results[idx]
        norm_val_query, scalar_val_query = create_eval_query(ex["model_response"], ex["ground_truth"], ex["question"])
        norm_val_queries.append(norm_val_query)
        scalar_val_queries.append(scalar_val_query)

    norm_val_responses = llm_infer(norm_val_queries)
    scalar_val_responses = llm_infer(scalar_val_queries)

    evaluations = {}
    for norm_val, scalar_val, (id, ex) in zip(norm_val_responses, scalar_val_responses, examples.items()):
        try:
            scalar_val = int(scalar_val)
        except:
            pass

        eval = {
            "question": ex["question"],
            "rationale": ex["rationale"],
            "predicted_answer": ex["model_response"],
            "correct_answer": ex["ground_truth"],
            "accuracy": scalar_val,
            "em_accuracy": 1 if scalar_val > 8 else 0,
            "gpt_eval": norm_val,
            "characteristic_breakdown": ex["characteristic_breakdown"],
        }
        evaluations[id] = eval
    
    return evaluations


def main(args):
    load_dotenv(ENV_PATH)

    with open(args.model_output_file, "r") as f:
        model_output = json.load(f)
        if args.num_to_append:
            model_output = model_output[-args.num_to_append:]

    eval_output = {
        "accuracy": {
            "mean": 0,
            "median": 0,
            "stdev": 0,
            "max": 0,
            "min": 0,
            "em_accuracy": 0
        },
        "evaluations": {},
    }

    eval_output["evaluations"] = send_eval_queries(model_output, x_shot_prompt=args.x_shot_prompt)
    Path(args.eval_output_file).parent.mkdir(parents=True, exist_ok=True)

    if args.num_to_append:
        with(open(args.eval_output_file, "r")) as f:
            old_eval_output = json.load(f)
        eval_output["evaluations"] = old_eval_output["evaluations"] + eval_output["evaluations"]

    valid_df = pd.read_csv(VALIDITY_CSV_PATH)
    valid_df.set_index(EXAMPLES, inplace=True)

    scores = []
    em_scores = []
    for id, example in eval_output["evaluations"].items():
        if args.validity == NO_INV and \
            valid_df.loc[id, TABLE_VALIDITY] == 0:
            continue

        if args.validity == ONLY_VAL and \
            (valid_df.loc[id, TABLE_VALIDITY] == 0 or pd.isna(valid_df.loc[id, TABLE_VALIDITY])):
            continue

        if isinstance(example["accuracy"], int):
            scores.append(example["accuracy"])
            em_scores.append(example["em_accuracy"])

    eval_output["accuracy"]["mean"] = mean(scores)
    eval_output["accuracy"]["median"] = median(scores)
    eval_output["accuracy"]["stdev"] = stdev(scores)
    eval_output["accuracy"]["max"] = max(scores)
    eval_output["accuracy"]["min"] = min(scores)
    eval_output["accuracy"]["em_accuracy"] = mean(em_scores)
    
    with open(args.eval_output_file, "w") as f:
        json.dump(eval_output, f, indent=4)


if __name__ == "__main__":
    args = evaluate_args()
    main(args)