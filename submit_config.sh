#!/bin/bash
#
#SBATCH --job-name=iwilds_training_default
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gpus=1

source .sherlock_env
if [ "$#" -eq 1 ]; then
    poetry run python holistic_reliability_evaluation/train.py --config $1
elif [ "$#" -eq 2 ]; then
    poetry run python holistic_reliability_evaluation/train.py --config $1 --seed $2
fi
