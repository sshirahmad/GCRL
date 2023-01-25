# Used for Domain Generalization Experiment

# PRETRAIN MODELS (v4 MLP)
GPU=0 # 0. Set GPU
exp="pretrain"

DEFAULTS="--decoupled_loss --coupling --no-rel_recon"
EVALUATION="--metrics accuracy"
dataset="v4"
model="mlp"
dset_type="test"
bs=64
z_dim=2
num_envs=5

epoch=250
for f_envs in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8"
do
    DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --dset_type $dset_type"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${z_dim}_cluster_${num_envs}.pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${z_dim}_cluster_${num_envs}.pth.tar" --seed $seed
done
