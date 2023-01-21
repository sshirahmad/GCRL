# PRETRAIN MODELS (v4 MLP)

GPU=2 # 0. Set GPU
EVALUATION="--metrics accuracy"
add_confidence=True
exp="pretrain"
domain_shifts='1-2-4-8-64'
dataset="eth"
model="lstm"
dset_type="test"
bs=64

epoch=700
best_ep=528 # change according to the last saved model
for shift in "8" "16" "32" "64"
do
    DATA="--dataset_name $dataset --domain_shifts $shift --add_confidence $add_confidence --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_${domain_shifts}_bk_1_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_False_epoch_${best_ep}.pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_${domain_shifts}_bk_1_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_False_epoch_${best_ep}.pth.tar" --seed $seed
done
