#!/bin/bash

#PBS -A starting_2023_110
#PBS -N EXPtorchrun
### Regular queue
#PBS -l walltime=0:20:00
#PBS -l nodes=2:ppn=1:gpus=1
#PBS -o \$PBS_JOBNAME\$PBS_JOBID.out
#PBS -e \$PBS_JOBNAME\$PBS_JOBID.err



nodes=( $( cat $PBS_NODEFILE ) )
echo nodes: $nodes
nodes_array=($nodes)
echo nodes_array: $nodes_array
head_node=${nodes_array[0]}
head_node_ip=$(ssh $head_node hostname -I | awk '{print $1}')

echo PBS_NUM_PPN = $PBS_NUM_PPN
echo PBS_NUM_GPUS = $PBS_NUM_GPUS

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd $PBS_O_WORKDIR

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
##module load foss/2023a
##module load vsc-mympirun
module load intel
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

mpirun -np $NUM_NODES -hostfile $PBS_NODEFILE \
torchrun \
--nnodes $NUM_NODES \
--nproc_per_node 1 \
--rdzv_id $JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
../multinode_torchrun.py 50 10

##--nproc_per_node 1 \  # corresponds to --gpus-per-task 
## --rdzv_endpoint $head_node_ip:29500 \
## --rdzv_endpoint $HOST_NODE_ADDR \
--nproc_per_node $NUM_TRAINERS \