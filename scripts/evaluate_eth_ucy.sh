# Used to Evaluate experiments on ETH-UCY environments

GPU=0
exp="pretrain"

DEFAULTS="--no-decoupled_loss --coupling --rel_recon"
EVALUATION="--metrics accuracy"
dataset="hotel"
model="lstm"
dset_type="test"
bs=64
z_dim=8
num_envs=5
epoch=1000
e_best=1000

DATA="--dataset_name $dataset --batch_size $bs --dset_type $dset_type"
seed=1
CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/GCRL_data_${dataset}_ds_0_bk_1_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_False_latentdim_${z_dim}_cluster_${num_envs}_epoch_${e_best}.pth.tar" --z_dim $z_dim --s_dim $z_dim --seed $seed
