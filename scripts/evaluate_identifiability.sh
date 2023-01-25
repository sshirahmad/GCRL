# Used for Domain Generalization Experiment

# PRETRAIN MODELS (v4 MLP)
GPU=0 # 0. Set GPU
exp="pretrain"

DEFAULTS="--decoupled_loss --coupling --no-rel_recon"
EVALUATION="--metrics accuracy"
dataset="v4"
model="mlp"
dset_type="val"
bs=64
f_envs='0.1-0.3-0.5'
DATA="--dataset_name $dataset --filter_envs $f_envs --batch_size $bs --reduceall 9000 --dset_type $dset_type"
z_dim=2
num_envs=5

epoch=250
paths=()
for seed in 1 2 3 4 5
do
     paths+=("models/$dataset/$exp/VCRL_data_${dataset}_ds_0_bk_20_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_True_latentdim_${z_dim}_cluster_${num_envs}.pth.tar")
done
CUDA_VISIBLE_DEVICES=$GPU python identifiability.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "${paths[@]}" --seed $seed
