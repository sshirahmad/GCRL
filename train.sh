#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --account=rrg-mcrowley
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=11
#SBATCH --job-name=crmf             
#SBATCH --output=crmf_%j.log
#SBATCH --mail-user=shayan.shirahmadi@gmail.com
#SBATCH --mail-type=ALL

python train.py