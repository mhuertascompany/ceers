#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --time=05:00:00
#SBATCH --job-name=florah
#SBATCH --output=%j.out
#SBATCH --error=%j.err


module load ml_py310

python3 florah_trainining_SFR.py