#!/bin/bash
#SBATCH --job-name=summarize-23
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551

# Load required modules
module load cuda/cuda-12.6 
module load python/3.10


# Upgrade pip
python3 -m pip install --no-index --upgrade pip --user

pip cache purge
pip install --no-index -r requirements.txt
pip install -r requirements.txt

# Run your Python script
python3 scripts2/retriever_2.py
