#!/bin/sh
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name GCN-Protein
#SBATCH --output=results/slurm-%A_%a.out
#SBATCH --array=0-24%25
#SBATCH --exclude=kathleencpu[05]

seed=$1
date=$2
nfolds=25
dataset="PROTEIN"
outputdir="results/GCN_${dataset}.${date}/"
mkdir $outputdir > /dev/null
filename="${outputdir}results_dictionary.pkl"

params=()
for ((i=0; i<$nfolds; i++)) do
    params[i]="--task graph --model GCN --dataset ${dataset} --filename ${filename} --seed ${seed} --foldindex ${i}"
done

srun python main.py ${params[$SLURM_ARRAY_TASK_ID]}