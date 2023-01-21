# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

dataset='eth' # 2. Set dataset
model="lstm"
batch_method="hom"
add_confidence=True
domain_shifts='1-2-4-8-64'
DATA="--dataset_name $dataset --add_confidence $add_confidence --domain_shifts $domain_shifts"
MODEL="--model_name $model"
DIR="--tfdir runs/$dataset/$exp/"
bs=64

e=300
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method"

for seed in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL --seed $seed &
done
CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5
