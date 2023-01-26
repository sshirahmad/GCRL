#! /usr/bin/env bash

dataset="v4"
exp='finetune'
finetune='all'

mkdir -p results/$dataset/$exp/$finetune
echo 'batches, split, envs, seed, ADE, FDE' | tee results/$dataset/$exp/$finetune/summary.csv

for filename in log/$dataset/$exp/$finetune/*.log; do
	cat ${filename} \
	| grep "Batches:\|Split:\|Envs:\|Seed:\|ADE:" \
	| sed "s/.*Batches: \([0-9]*\).*/\1,/g" \
	| sed "s/.*Split: \([a-z.]*\).*/\1,/g" \
	| sed "s/.*Envs: \([0-9.-]*\).*/\1,/g" \
	| sed "s/.*Seed: \([0-9]*\).*/\1,/g" \
	| sed "s/.*ADE: \([0-9.]*\).*FDE: \([0-9.]*\).*/\1, \2 /g" \
	| paste -d " " - - - - - \
	| tee -a results/$dataset/$exp/$finetune/summary.csv
done

python visualize.py --exp $exp --finetune $finetune --dataset_name $dataset