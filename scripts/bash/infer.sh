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

max_new_tokens=1024
max_length=1048576

# Loop through the JSON array
jq -c '.[]' "$LLM_FILE" | while read -r line; do
    # Extract fields from JSON
    model=$(echo "$line" | jq -r '.model')
    name=$(echo "$line" | jq -r '.name')

    if [ api == 0 ]; then
        tokenizer=$(echo "$line" | jq -r '.tokenizer')
        task=$(echo "$line" | jq -r '.task')
        remote=$(echo "$line" | jq -r '.remote')
    fi

    cmd=(
        python scripts/llm/main.py
        --model "$model"
        --name "$name"
        --max_new_tokens $max_new_tokens
        --max_length $max_length
        --dataset_path "data/datasets/$dataset"
        --x_shot_prompt "$x_shot_prompt"
        --reasoning_type "$reasoning_type"
        --num_to_append "$num_to_append"
    )

    output_file=iclr_v2_model_output/$dataset/$reasoning_type/$name/${x_shot_prompt}_output.json

    if [ api != 0 ]; then
        echo "Running inference on ${name}"
        cmd+=(
            --output_file "$output_file"
            --is_api_model
        )
    else
        echo "Running inference on ${name}"
        cmd+=(--tokenizer_name "$tokenizer")
        cmd+=(--task "$task")
        cmd+=(--output_file "$output_file")
        if [[ "$remote" == "true" ]]; then
            cmd+=(--remote)
        fi
    fi

    echo  "${cmd[@]}"
    # Execute the command
    "${cmd[@]}"

done
