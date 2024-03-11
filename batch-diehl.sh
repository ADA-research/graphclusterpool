#!/bin/sh
#SBATCH --mem-per-cpu=3000
#SBATCH --job-name DIEHL-R
#SBATCH --output=results/Rerun/Diehl/slurm-%A_%a.out
#SBATCH --array=0-99%100
#SBATCH --partition=Kathleenhigh,Kathleenlow

dataset=$1
seed=$2
date=$3
nfolds=100
outputdir="results/Rerun/Diehl/${dataset}/GCN_${dataset}.${date}/"
mkdir $outputdir > /dev/null
filename="${outputdir}results_dictionary.pkl"

params=()
for ((i=0; i<$nfolds; i++)) do
    params[i]="--task graph --dataset ${dataset} --filename ${filename} --seed ${seed} --foldindex ${i} --rerun diehl"
done

srun python main.py ${params[$SLURM_ARRAY_TASK_ID]}