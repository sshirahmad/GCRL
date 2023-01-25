# Used for Domain Adaptation Experiment

GPU=0
exp='finetune'

DEFAULTS="--decoupled_loss --coupling --no-rel_recon"
EVALUATION='--metrics accuracy'
model="mlp"
f_envs='0.6'
dataset='v4'
bs=64
DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs"

for finetune in 'all' 'weight+s'
do
    for reduce in  0 64 128 192 256 320 384
    do
        for seed in 1 2 3 4
        do
            CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --reduce $reduce --finetune $finetune --resume "models/$dataset/$exp/$finetune/$seed/model_t${reduce}.pth.tar" --seed $seed &
        done
        seed=5
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --reduce $reduce --finetune $finetune --resume "models/$dataset/$exp/$finetune/$seed/model_t${reduce}.pth.tar" --seed $seed
    done
done
