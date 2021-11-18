#!/bin/bash
#SBATCH -J hubert_pretrain
#SBATCH -o slurm_logs/hubert_pretrain_%j.out
#SBATCH -e slurm_logs/hubert_pretrain_%j.err
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=ssl_disentangle
CONDA_ROOT=/nobackup/users/$(whoami)/espnet/tools/conda

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
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

# (143553)
srun ./run_pretrain_v01.sh
# (141758)
# ./run.sh --stage 10 --stop-stage 10
# (141769)
# ./run.sh --stage 10 --stop-stage 10 --exp-name hubert_1_nopraat
# (141886) 
# ./run.sh --stage 10 --stop-stage 10 --exp-name hubert_1_praat
echo "Run completed at:- "
date
