#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/outputs_final/logs/%x_%j.out
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
H=/home/manik/.cache/huggingface
C=default
case $MODEL in
  medgemma) E=gdino; U=1; H=/home/manik/pranjali/Aditya_project/.cache/huggingface; C=project ;;
  maira2) E=maira2; U=0; H=/home/manik/pranjali/Aditya_project/.cache/huggingface; C=project ;;
  chexagent) E=chexagent; U=0; H=/home/manik/pranjali/Aditya_project/.cache/huggingface; C=project ;;
  radvlm) E=radvlm; U=0 ;;
  chex) E=chex; U=1 ;;
esac
conda activate $E
if [ "$U" = 1 ]; then export PYTHONNOUSERSITE=1; fi
export HF_HOME=$H
export CFG=$C
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
if [ "$J0" = 1 ]; then
  python hpc/run_shard.py --model $MODEL --mode uniform --tag j0a --subset j0 || exit 1
  python hpc/run_shard.py --model $MODEL --mode uniform --tag j0b --subset j0 || exit 1
fi
python hpc/run_shard.py --model $MODEL --mode $MODE --tag $TAG --nshards $NSHARDS --shards ${SHARDS//+/,} --limit ${LIMIT:-0}
if [ -n "$MODE2" ]; then
  python hpc/run_shard.py --model $MODEL --mode $MODE2 --tag $TAG2 --nshards $NSHARDS --shards ${SHARDS//+/,} --limit ${LIMIT:-0}
fi
