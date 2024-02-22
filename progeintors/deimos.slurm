#!/bin/bash
#SBATCH --account=mhuertas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --time=05:00:00
#SBATCH --job-name=florah
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mhuertas@iac.es

module load ml_py310

python3 florah_trainining_SFR.py