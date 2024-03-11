declare -A seedmap=( ["batch-collab.sh"]="80417196" ["batch-protein.sh"]="92230607" ["batch-reddit12k.sh"]="36441034" ["batch-redditb.sh"]="20855121" )
#seed=$(($RANDOM*$RANDOM))
seed=${seedmap[$1]}
printf -v date '%(%Y-%m-%d.%H:%M:%S)T' -1
sbatch $1 $seed $date