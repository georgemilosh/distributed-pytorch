#!/bin/bash -l

#PBS -A starting_2023_110
#PBS -N multigpu
#PBS -m abe
### Regular queue
#PBS -l walltime=0:20:00
#PBS -l nodes=1:gpus=4
#PBS -o \$PBS_JOBNAME\$PBS_JOBID.out
#PBS -e \$PBS_JOBNAME\$PBS_JOBID.err


cd $PBS_O_WORKDIR

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
##mamba activate torch
python ../multigpu.py 50 10