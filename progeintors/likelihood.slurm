#!/bin/bash
#SBATCH --ntasks=1                  # 1 proceso principal
#SBATCH --cpus-per-task=8           # 8 cores para paralelismo
#SBATCH --mem=64G                   # memoria grande para muchos datos
#SBATCH --gres=gpu:1                # <-- NUEVO: Solicitar 1 GPU
#SBATCH --time=12:00:00             # ajusta según cuánto dure tu job
#SBATCH --job-name=likelihood_cosmos
#SBATCH --output=/scratch/lmarrero-ext/likelihood_COSMOS_SFR/logs/%j.out             # salida 
#SBATCH --error=/scratch/lmarrero-ext/likelihood_COSMOS_SFR/logs/%j.err              # errores

# === Cargar entorno Python del cluster ===
module load ml_py310              

# === Configurar threads para librerías matemáticas ===
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# === Ejecutar tu script Python ===
python3 likelihood_COSMOS_SFR_optimized_SCRIPT.py