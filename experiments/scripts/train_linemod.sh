#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONWARNINGS="ignore"

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=3

start=`date +%s`

python3 ./tools/train.py --nepoch 101 --dataset linemod\
    --dataset_root ./datasets/linemod/Linemod_preprocessed

end=`date +%s`

runtime=$((end-start))
echo $runtime