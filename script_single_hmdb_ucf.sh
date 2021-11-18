#!/bin/sh

#SBATCH --job-name=single_hmdb_ucf
#SBATCH --account=tud01
#SBATCH --mem=16384
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --output=res_single_hmdb_ucf.out

module load conda/2
module load cuda/11.2

source activate /ichec/home/users/kaiqiang/py39

echo "Single GAN for HMDB51 and UCF101."

time python ./scripts/run_hmdb51_tfvaegan.py
time python ./scripts/run_ucf101_tfvaegan.py