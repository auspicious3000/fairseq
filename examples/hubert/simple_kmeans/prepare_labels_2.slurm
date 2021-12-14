#!/bin/bash
#SBATCH -J hubert_pretrain
#SBATCH -o /nobackup/users/yangzhan/ssl-disentangle/logs/prepare_labels_%j.out
#SBATCH -e /nobackup/users/yangzhan/ssl-disentangle/logs/prepare_labels_%j.err
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/kzqian
PYTHON_VIRTUAL_ENVIRONMENT=opence_v04
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=/nobackup/users/yangzhan/ssl-disentangle/logs/nodelist.$
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

srun --nodes=1 --ntasks-per-node=1 ./run_prepare_data_batch_1.sh 802 841 &
srun --nodes=1 --ntasks-per-node=1 ./run_prepare_data_batch_1.sh 842 881 &
srun --nodes=1 --ntasks-per-node=1 ./run_prepare_data_batch_1.sh 882 921 &
srun --nodes=1 --ntasks-per-node=1 ./run_prepare_data_batch_1.sh 922 961 &
wait

echo "Run completed at:- "
date
