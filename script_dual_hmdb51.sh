#!/bin/sh

#SBATCH --job-name=hmdb_gzsl_od_dual_free_600
#SBATCH --account=tucom002c
#SBATCH --mem=65536
#SBATCH --partition GpuQ
#SBATCH --nodes 1
#SBATCH --time=02-00
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=kaiqiang.x.huang@mytudublin.ie
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --output=res_dual_hmdb51_gzsl_od_dual_free_600_sp_17_30.out

module load conda/2
module load cuda/11.2

source activate /ichec/work/tucom002c/py39

echo "GZSL OD: Dual GAN with FREE for HMDB51 (multi-GPU)"
echo "The number of syn_data is 600"

time python ./dual/run_hmdb51_tfvaegan_dual.py