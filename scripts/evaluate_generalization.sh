# PRETRAIN MODELS (v4 MLP)

GPU=2 # 0. Set GPU
EVALUATION="--metrics accuracy"
exp="pretrain"
dataset="v4"
model="mlp"
dset_type="test"
bs=64

epoch=400
for f_envs in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_epoch_${epoch}.pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_all.py $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_epoch_${epoch}.pth.tar" --seed $seed
done
