#!/bin/bash
#SBATCH --ntasks=1                  # 1 proceso principal
#SBATCH --cpus-per-task=2           # 2 cores
#SBATCH --mem=64G                   # memoria grande para muchos datos
#SBATCH --gres=gpu:1                # <-- NUEVO: Solicitar 1 GPU
#SBATCH --time=1-00:00:00           # Ajustado a 1 día (24 horas)
#SBATCH --job-name=likelihood_cosmos
#SBATCH --output=/scratch/lmarrero-ext/likelihood_COSMOS_SFR/logs/%j.out             # salida 
#SBATCH --error=/scratch/lmarrero-ext/likelihood_COSMOS_SFR/logs/%j.err              # errores

# === Cargar entorno Python del cluster ===
module load ml_py310   
module load python/3.12_b1125
           

# Force the cluster to use exactly 1 thread for data processing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# === Ejecutar tu script Python ===
python3 likelihood_COSMOS_v2.py