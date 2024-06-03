declare -A seedmap=( ["batch-collab.sh"]="80417196" ["batch-protein.sh"]="92230607" ["batch-reddit12k.sh"]="36441034" ["batch-redditb.sh"]="20855121" ["batch-imdbb.sh"]="368472694" ["batch-imdbm.sh"]="668028379" ["batch-reddit5k.sh"]="305906664" ["batch-nci1.sh"]="107236008" )
seed=${seedmap[$1]}
printf -v date '%(%Y-%m-%d.%H:%M:%S)T' -1
sbatch $1 $seed $date