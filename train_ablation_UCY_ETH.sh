# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

DEFAULTS="--no-decoupled_loss --coupling --rel_recon --lr_scheduler"
dataset='eth' # 2. Set dataset
model="lstm"
batch_method="hom"
DATA="--dataset_name $dataset"
MODEL="--model_name $model"
DIR="--tfdir runs/$dataset/$exp/"
bs=64
num_envs=2
dim=8
e=300
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method"

CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $DIR $MODEL --z_dim $dim --s_dim $dim --seed 1 --num_envs $num_envs