#!/bin/bash
#
#SBATCH --job-name=iwilds_training_default
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH -p gpu
#SBATCH --gpus=1

source .sherlock_env
poetry run python holistic_reliability_evaluation/train.py --config $1