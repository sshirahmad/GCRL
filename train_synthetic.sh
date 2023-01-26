# USed to train models on v4 dataset

## General parameters
GPU=0 # 1. Set GPU
exp='pretrain'

DEFAULTS="--decoupled_loss --coupling --no-rel_recon --no-lr_scheduler"
dataset='v4' # 2. Set dataset
model="mlp"
f_envs='0.1-0.3-0.5'
batch_method="het"
DATA="--dataset_name $dataset --filter_envs $f_envs --reduceall 9000"
MODEL="--model_name $model"
DIR="--tfdir runs/$dataset/$exp/"
bs=64

e=250
TRAINING="--num_epochs $e --batch_size $bs --batch_method $batch_method --best_k 20"

for seed in 1 2 3 4
do  
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $DIR $MODEL --seed $seed &
done
CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $DIR $MODEL --seed 5
