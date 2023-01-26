# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

DEFAULTS="--no-decoupled_loss --coupling --rel_recon --lr_scheduler"
model="lstm"
batch_method="hom"
MODEL="--model_name $model"
bs=64
dim=8
e=300
seed=1
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method"

for dataset in "eth" "hotel" "univ" "zara1"
do
    DIR="--tfdir runs/$dataset/$exp/"
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $TRAINING $DIR $MODEL --dataset_name $dataset --seed $seed --z_dim $dim --s_dim $dim &
done
DIR="--tfdir runs/zara2/$exp/"
CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $TRAINING $DIR $MODEL --seed 1 --z_dim $dim --s_dim $dim --dataset_name "zara2"