#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONWARNINGS="ignore"

#export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=2,1
#export CUDA_VISIBLE_DEVICES=2,1,0


start=`date +%s`
    
#python train.py
python test.py --model_save_path ./trained_models/model_97_0.11637805987335742.pth

end=`date +%s`

runtime=$((end-start))
echo $runtime

