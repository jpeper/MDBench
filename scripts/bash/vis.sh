#!/bin/bash

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 <dataset> <validity>"
    exit 1
fi

dataset=${1:-"main"}
validity=${2:-"all"}
eval_folder=${3:-"iclr_v2_eval_output"}
output_folder=${4:-"iclr_v2_viz_output"}

python scripts/eval/visualize.py -dcb -r -dd $dataset -v $validity -e $eval_folder -o $output_folder

for characteristic in "multi-hop" "temporal" "numeric" "information_aggregation" "soft_reasoning"; do
    for x_shot_prompt in zero-shot; do
    #for x_shot_prompt in zero-shot zero-shot-CoT few-shot few-shot-CoT; do
        python scripts/eval/visualize.py -cb -p $x_shot_prompt -c $characteristic -dd $dataset -v $validity -e $eval_folder -o $output_folder
    done
    python scripts/eval/visualize.py -ocb -c $characteristic -dd $dataset -v $validity -e $eval_folder -o $output_folder
done
