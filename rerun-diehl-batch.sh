declare -A seedmap=(["COLLAB"]="42",["PROTEIN"]="42",["REDDIT-MULTI-12K"]="42",["REDDIT-BINARY"]="42")

printf -v date '%(%Y-%m-%d.%H:%M:%S)T' -1

seed = seedmap[$1]

sbatch batch-diehl.sh $1 $seed $date