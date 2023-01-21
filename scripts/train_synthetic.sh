# PRETRAIN

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

dataset='v4' # 2. Set dataset
model="mlp"
dc=True
f_envs='0.1-0.3-0.5'
batch_method="het"
rel_recon=False
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 9000"
MODEL="--model_name $model"
DIR="--tfdir runs/$dataset/$exp/"
bs=64

e=700
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method --decoupled_loss $dc --best_k 20 --rel_recon $rel_recon"

for seed in 1 2 3 4
do  
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL --seed $seed &
done
CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $DIR $MODEL $USUAL --seed 5
