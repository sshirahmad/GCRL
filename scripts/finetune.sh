# Used for Finetuning models on v4 dataset
exp='finetune'
GPU=0 # 0. Set GPU

# data
DEFAULTS="--decoupled_loss --coupling --no-rel_recon --no-lr_scheduler"
model="mlp"
f_envs='0.6'
batch_method="het"
dataset='v4'
DATA="--dataset_name $dataset --filter_envs $f_envs"
MODEL="--model_name $model"
bs=64
ldim=2
num_envs=5
finetune='all'

## TO CHANGE DEPENDING ON PREVIOUS STEPS
steps=100 # number of finetuning steps
epoch=250


for seed in 1 2 3 4 5
do
    # pretrained model
    DIR="--tfdir runs/$dataset/$exp/$finetune/$seed"

    TRAINING="--num_epochs $steps --batch_size $bs --batch_method $batch_method --finetune $finetune --best_k 20"

    for reduce in 0 64 128 192 256 320
    do
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $MODEL $DIR --reduce $reduce --seed $seed --resume "models/$dataset/pretrain/GCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${ldim}_cluster_${num_envs}.pth.tar" &
    done
    CUDA_VISIBLE_DEVICES=$GPU python train.py $DEFAULTS $DATA $TRAINING $MODEL $DIR --reduce 384 --seed $seed --resume "models/$dataset/pretrain/GCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${ldim}_cluster_${num_envs}.pth.tar"
done
