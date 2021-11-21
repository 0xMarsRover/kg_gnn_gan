#!/bin/sh

#SBATCH --job-name=dual_rf_hmdb
#SBATCH --account=tud01
#SBATCH --mem=32768
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --output=res_dual_hmdb51.out

module load conda/2
module load cuda/11.2

source activate /ichec/home/users/kaiqiang/py39

echo "Dual GAN with different classifier for HMDB51 (multi-GPU)"

time python ./dual/run_hmdb51_tfvaegan_dual.py