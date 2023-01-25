# Used for Robustness and Domain Generalization Results

dataset="v4"
exp='pretrain'

mkdir -p results/$dataset/$exp
echo 'split, shifts, envs, seed, ADE, FDE' | tee results/$dataset/$exp/summary.csv

for filename in log/$dataset/$exp/*.log; do
	cat ${filename} \
	| grep "Split:\|Shifts:\|Envs:\|Seed:\|ADE:" \
	| sed "s/.*Split: \([a-z.]*\).*/\1,/g" \
	| sed "s/.*Shifts: \([0-9]*\).*/\1,/g" \
	| sed "s/.*Envs: \([0-9.-]*\).*/\1,/g" \
	| sed "s/.*Seed: \([0-9]*\).*/\1,/g" \
	| sed "s/.*ADE: \([0-9.]*\).*FDE: \([0-9.]*\).*/\1, \2 /g" \
	| paste -d " " - - - - - - \
	| tee -a results/$dataset/$exp/summary.csv
done

python visualize.py --exp $exp --dataset_name $dataset