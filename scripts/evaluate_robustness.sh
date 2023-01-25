# Used for Robustness Experiment

GPU=0
exp="pretrain"

DEFAULTS="--no-decoupled_loss --coupling --rel_recon"
EVALUATION="--metrics accuracy"
domain_shifts='1-2-4-8-64'
dataset="eth"
model="lstm"
dset_type="test"
bs=64
z_dim=8
num_envs=5
epoch=300

for shift in "8" "16" "32" "64"
do
    DATA="--dataset_name $dataset --domain_shifts $shift --add_confidence --batch_size $bs --dset_type $dset_type"
    seed=1
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_model.py $DEFAULTS $DATA $EVALUATION --model_name $model --resume "models/$dataset/$exp/GCRL_data_${dataset}_ds_${domain_shifts}_bk_1_ns_10_ep_${epoch}_seed_${seed}_cl_True_dc_False_latentdim_${z_dim}_cluster_${num_envs}_epoch_${epoch}.pth.tar" --z_dim $z_dim --s_dim $z_dim --seed $seed
done
