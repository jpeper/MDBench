#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-15:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=11500m
#SBATCH --account=wangluxy1
#SBATCH --mail-user=qwzhao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=mddg_toy

# conda
module load python3.10-anaconda/2023.03
pushd /home/qwzhao/src/mddg

source /sw/pkgs/arc/python3.10-anaconda/2023.03/bin/activate
conda activate mddg


bash scripts/bash/toy.sh run llama-3/small_llama 0
bash scripts/bash/toy.sh eval llama-3/small_llama 0
bash scripts/bash/vis.sh