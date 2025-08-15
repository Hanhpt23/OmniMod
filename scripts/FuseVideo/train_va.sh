#!/bin/bash

#SBATCH --job-name=train    
#SBATCH --output=gpu_output/Train-JOB_ID_%j-%N.log 

#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --mem=150G
#SBATCH --partition=gpu2

echo "Job ID: $SLURM_JOBID" 
echo "Node names: $SLURM_JOB_NODELIST"
echo "Notes: Training FuseVideo"
CUDA_VISIBLE_DEVICES=0
nvidia-smi 

export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

torchrun --nproc_per_node 2 train.py \
      --cfg-path train_videoaudio_configs/train_videoaudio.yaml\
      --cfg-eval-path eval_videoaudio_configs/evaluate_videoaudio.yaml\
      --eval-dataset audiovideo_val
