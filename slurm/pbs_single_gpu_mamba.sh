#!/bin/bash -l

#PBS -A starting_2023_110
#PBS -N single_gpu_mamba
#PBS -m abe
### Regular queue
#PBS -l walltime=0:20:00
#PBS -l nodes=1:gpus=1
#PBS -o \$PBS_JOBNAME\$PBS_JOBID.out
#PBS -e \$PBS_JOBNAME\$PBS_JOBID.err


cd $PBS_O_WORKDIR

mamba activate torch
python ../single_gpu.py 50 10