declare -A seedmap=( ["COLLAB"]="80417196" ["PROTEIN"]="92230607" ["REDDIT-MULTI-12K"]="36441034" ["REDDIT-BINARY"]="20855121" )

printf -v date '%(%Y-%m-%d.%H:%M:%S)T' -1

seed_value=${seedmap[$1]}

sbatch batch-diehl.sh $1 $seed_value $date