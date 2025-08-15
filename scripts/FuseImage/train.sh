#!/bin/bash

#SBATCH --job-name=train    
#SBATCH --output=gpu_output_image/Train-JOB_ID_%j-%N.log 

#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  

echo "Job ID: $SLURM_JOBID" 
echo "Node names: $SLURM_JOB_NODELIST"
echo "Notes: Training FuseVideo"
CUDA_VISIBLE_DEVICES=2
nvidia-smi 

export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

torchrun --nproc_per_node 2 train.py \
      --cfg-path train_configs/train_image.yaml\
      --cfg-eval-path eval_configs/evaluate_image.yaml\
      --eval-dataset image_val
