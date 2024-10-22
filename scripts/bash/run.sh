#!/bin/bash

if [ "$#" -lt 2 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <script> <llm_file> [<dataset>] [<validity>] [<api>] [<num_to_append>]"
    exit 1
fi

# Activate your conda environment
CONDA_ENV_NAME="kaa"  # Replace with your conda environment name

# Check if Conda is initialized and activate the environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

script=$1
llm_file=$2
dataset=${3:-"main"}
validity=${4:-"all"}
api=${5:-1}
num_to_append=${6:-0}  # Default value of 0 if not provided

for reasoning_type in documents table; do
    #for x_shot_prompt in zero-shot; do
    for x_shot_prompt in zero-shot zero-shot-CoT few-shot few-shot-CoT; do
    #dfor x_shot_prompt in zero-shot-CoT few-shot-CoT; do
        echo scripts/bash/$script.sh $llm_file $api $validity $dataset $x_shot_prompt $reasoning_type $num_to_append
        bash scripts/bash/$script.sh $llm_file $api $validity $dataset $x_shot_prompt $reasoning_type $num_to_append
    done
done

