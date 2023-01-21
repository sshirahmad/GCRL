# FINE TUNE
exp='finetune'
GPU=0 # 0. Set GPU

# data
model="mlp"
dc=True
f_envs='0.6'
batch_method="het"
rel_recon=False
dataset='v4'
DATA="--dataset_name $dataset --filter_envs $f_envs"
MODEL="--model_name $model"
bs=64

## TO CHANGE DEPENDING ON PREVIOUS STEPS
steps=100 # number of finetuning steps
epoch=400

for finetune in 'all' 'weights+s'
do
    for seed in 1 2 3 4 5
    do
        # pretrained model
        DIR="--tfdir runs/$dataset/$exp/$finetune/$seed"

        TRAINING="--num_epochs $steps --batch_size $bs --batch_method $batch_method --finetune $finetune --decoupled_loss $dc --best_k 20 --rel_recon $rel_recon"

        for reduce in 64 128 192 256 320
        do
            CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce $reduce --seed $seed --resume "models/$dataset/pretrain/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_epoch_${epoch}.pth.tar" &
        done
        CUDA_VISIBLE_DEVICES=$GPU python train.py $DATA $TRAINING $MODEL $DIR --reduce 384 --seed $seed --resume "models/$dataset/pretrain/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_epoch_${epoch}.pth.tar"
    done
done