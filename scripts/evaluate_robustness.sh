# PRETRAIN MODELS (v4 MLP)

GPU=0
EVALUATION="--metrics accuracy"
add_confidence=True
exp="pretrain"
domain_shifts='1-2-4-8-64'
dataset="eth"
model="lstm"
dset_type="test"
bs=64

epoch=300
for shift in "8" "16" "32" "64"
do
    DATA="--dataset_name $dataset --domain_shifts $shift --add_confidence $add_confidence --batch_size $bs --dset_type $dset_type"
    seed=1
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_${domain_shifts}_bk_1_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_False_epoch_${epoch}.pth.tar" --z_dim 8 --s_dim 8 --seed $seed
done
