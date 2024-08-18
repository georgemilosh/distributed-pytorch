#!/bin/bash

#PBS -A starting_2023_110
#PBS -N torchrun_debug
### Regular queue
#PBS -l walltime=0:15:00
#PBS -l nodes=2:ppn=1:gpus=1
#PBS -o \$PBS_JOBNAME\$PBS_JOBID.out
#PBS -e \$PBS_JOBNAME\$PBS_JOBID.err

nodes=( $( cat $PBS_NODEFILE ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(ssh $head_node hostname -I | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd $PBS_O_WORKDIR

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load vsc-mympirun
##module load intel
##mamba activate torch

# Calculate the number of processors per node
NUM_TRAINERS=$(($PBS_NP / $PBS_NUM_NODES))
NUM_NODES=$PBS_NUM_NODES
JOB_ID=$PBS_JOBID
HOST_NODE_ADDR=$(hostname)

echo JOB_ID = $JOB_ID
echo NUM_TRAINERS = $NUM_TRAINERS
echo NUM_NODES = $NUM_NODES
echo HOST_NODE_ADDR = $HOST_NODE_ADDR
echo PBS_NODEFILE = $PBS_NODEFILE

mympirun --hybrid $NUM_NODES \
torchrun \
--nnodes $NUM_NODES \
--nproc_per_node $NUM_TRAINERS \
--rdzv_id $JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $HOST_NODE_ADDR \
../multinode_torchrun.py 50 10

##--nproc_per_node 1 \  # corresponds to --gpus-per-task 
## --rdzv_endpoint $head_node_ip:29500 \