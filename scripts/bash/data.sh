#!/bin/bash

python scripts/dataprocess/create_dataset.py -l data/raw_data/v3_tuesday_50/ -o
python scripts/dataprocess/create_dataset.py -l data/raw_data/v4_tuesday_two_prompts/
python scripts/dataprocess/create_dataset.py -l data/raw_data/v4_tuesday_two_prompts_reversed/
python scripts/dataprocess/create_dataset.py -l data/raw_data/v4_tuesday_two_prompts_reversed_v4/

python scripts/dataprocess/create_toy_dataset.py