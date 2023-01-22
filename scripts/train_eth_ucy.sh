# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

DEFAULTS="--no_decoupled_loss --coupling --rel_recon --lr_scheduler"
dataset='eth' # 2. Set dataset
model="lstm"
batch_method="hom"
domain_shifts='1-2-4-8-64'
DATA="--dataset_name $dataset --add_confidence --domain_shifts $domain_shifts"
MODEL="--model_name $model"
DIR="--tfdir runs/$dataset/$exp/"
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method"
bs=64
dim=8
e=300

CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $DIR $MODEL --z_dim $dim --s_dim $dim --seed 1
