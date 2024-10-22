#!/bin/bash

if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <LLM_file> <api> <validity> <dataset> <x_shot_prompt> <reasoning_type> [<num_to_append>]"
    exit 1
fi

LLM_FILE=model_configs/${1}.json
api=$2
validity=${3:-"all"}
dataset=$4
x_shot_prompt=$5
reasoning_type=$6
num_to_append=${7:-0}  # Default value of 0 if not provided

# Loop through the JSON array
jq -c '.[]' "$LLM_FILE" | while read -r line; do
    # Extract fields from JSON
    model=$(echo "$line" | jq -r '.model')
    name=$(echo "$line" | jq -r '.name')

    echo "Evaluating ${name}"

    output_file=$dataset/$reasoning_type/$name/${x_shot_prompt}

    echo $output_file

    cmd=(
        python scripts/eval/evaluate.py
        --num_to_append $num_to_append
        --model_output_file iclr_v2_model_output/${output_file}_output.json
        --eval_output_file iclr_v2_eval_output/$validity/${output_file}_score.json
        --validity $validity
        --x_shot_prompt $x_shot_prompt
    )

    echo ${cmd[@]}
    "${cmd[@]}"

done