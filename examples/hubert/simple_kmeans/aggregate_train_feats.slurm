#!/bin/bash -x
#SBATCH -J hubert_pretrain
#SBATCH -o /gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/logs/prepare_labels_%j.out
#SBATCH -e /gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/logs/prepare_labels_%j.err
#SBATCH --time=6:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:6
#SBATCH --nodes=16
#SBATCH --cpus-per-task=80
#SBATCH --exclusive

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=opence_v01
CONDA_ROOT=/gpfs/u/scratch/LANG/LANGkzhq/anaconda3_202011

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=/gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/logs/nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

for node in $(seq 0 15); do
srun --nodes=1 --ntasks-per-node=1 python cumsum_mmap.py $node &
done
wait

echo "Run completed at:- "
date