#!/bin/sh

#SBATCH --job-name=ucf_gzsl_od_dual_600
#SBATCH --account=tucom002c
#SBATCH --mem=65536
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --output=res_dual_ucf101_gzsl_od_600_sp_7_10.out

module load conda/2
module load cuda/11.2

source activate /ichec/work/tucom002c/py39

echo "GZSL OD: Dual GAN without FREE for UCF101 (multi-GPU)"
echo "The number of syn_data is 600"

time python ./dual/run_ucf101_tfvaegan_dual.py