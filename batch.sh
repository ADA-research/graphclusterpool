#!/bin/sh
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name GCNpaper
srun python main.py --task graph --model GCN --dataset REDDIT-MULTI