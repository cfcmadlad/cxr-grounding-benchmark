R=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
M=$1; MODE=$2; TAG=$3; NSH=$4; SHARDS=$5; J0=$6; TL=$7; PART=$8; DEP=$9
sbatch --parsable -p $PART -J ${TAG}_${M}_${MODE}_${SHARDS//+/-} -t $TL ${DEP:+--dependency=afterany:$DEP} --export=ALL,MODEL=$M,MODE=$MODE,TAG=$TAG,NSHARDS=$NSH,SHARDS=$SHARDS,J0=$J0,LIMIT=0 $R/hpc/job.sbatch
