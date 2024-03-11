declare -A seedmap=( ["COLLAB"]="80417196" ["PROTEIN"]="92230607" ["REDDIT-MULTI-12K"]="36441034" ["REDDIT-BINARY"]="20855121" ["IMDB-BINARY"]="368472694" ["IMDB-MULTI"]="668028379" ["REDDIT-MULTI-5K"]="305906664" ["NCI1"]="107236008" )

printf -v date '%(%Y-%m-%d.%H:%M:%S)T' -1

seed_value=${seedmap[$1]}

sbatch batch-xu.sh $1 $seed_value $date