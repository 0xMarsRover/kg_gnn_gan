#!/bin/sh

#SBATCH --job-name=dual_ucf101
#SBATCH --account=tud01
#SBATCH --mem=32768
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --output=res_dual_ucf101.out

module load conda/2
module load cuda/11.2

source activate /ichec/home/users/kaiqiang/py39

echo "Dual GAN for HMDB51"

time python ./dual/run_ucf101_tfvaegan_dual.py