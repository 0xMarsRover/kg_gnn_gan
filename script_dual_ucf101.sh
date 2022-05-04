#!/bin/sh

#SBATCH --job-name=ucf_gzsl_od_600
#SBATCH --account=tud01
#SBATCH --mem=65536
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --output=res_ucf101_600.out

module load conda/2
module load cuda/11.2

source activate /ichec/home/users/kaiqiang/py39

echo "Single GAN for UCF101"
echo "The number of syn_data is 600"

time python ./scripts/run_ucf101_tfvaegan.py