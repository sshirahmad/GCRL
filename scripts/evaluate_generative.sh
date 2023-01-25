# Used for Best_N

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
for N in "20" "30" "40" "50" "60" "70" "80" "90" "100"
do
    DATA="--dataset_name $dataset --filter_envs 0.6 --batch_size $bs --dset_type $dset_type --best_k $N"
    for seed in 1 2 3 4
    do  
        CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/GCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${z_dim}_cluster_${num_envs}.pth.tar" --seed $seed &
    done
    seed=5
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/GCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${z_dim}_cluster_${num_envs}.pth.tar" --seed $seed
done
