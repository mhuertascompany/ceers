#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G                   # memoria grande para muchos datos
#SBATCH --gres=gpu
#SBATCH --time=23:00:00
#SBATCH --job-name=diff_florah
#SBATCH --output=/scratch/lmarrero-ext/florah_eval/logs/%j.out             # salida 
#SBATCH --error=/scratch/lmarrero-ext/florah_eval/logs/%j.err              # errores


module load ml_py310   
module load python/3.12_b1125

python3 differences.py