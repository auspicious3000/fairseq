#!/bin/bash
#SBATCH -J trainulm
#SBATCH -o trainulm_%j.out
#SBATCH -e trainulm_%j.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --mem=0

##SBATCH --cpus-per-task=10

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=ssl_disentangle
CONDA_ROOT=/nobackup/users/heting/espnet/tools/conda

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


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

# ckpt_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
# model_name="hubert"
# FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature.py"

ckpt_path="/nobackup/users/kzqian/ssl-disentangle/baseline/checkpoints/checkpoint_best.pt"
model_name="hubert_base"
FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature.py"

# ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
# model_name="hubert_v01_1"
# FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature_spk.py"

# Model Independent Setting
train_dataset="librilight6k_chunk"
km_dataset="librispeech100"


# t="50 100 200"
layer=8

if [ 0 -eq 0 ]; then
    vocab_size=100

    label_dir="$(pwd)/manifest/${train_dataset}/${model_name}_l${layer}_v${vocab_size}"
    data_dir="${label_dir}/bin"
    
    save_dir="$(pwd)/outputs/ulm_${model_name}_${train_dataset}_l${layer}_v${vocab_size}_8p"

    # rm -rf ${data_dir}
    # mkdir -p ${data_dir}
    # srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 5 fairseq-preprocess --only-source \
    #     --trainpref ${label_dir}/train.txt --validpref ${label_dir}/dev.txt --testpref ${label_dir}/test.txt \
    #     --destdir ${data_dir} --workers 5

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    PORT=54321
    init_method="tcp://${MASTER_ADDR}:${PORT}"
    echo "Params: data_dir  (${data_dir})"
    echo "Params: save_dir  (${save_dir})"
    echo "Params: init_method  (${init_method})"
    echo "Train ULM..."
    mkdir -p ${save_dir}
    if [[ -f "${save_dir}/train.log" ]]; then
        n_prev=$(ls -1q ${save_dir} | grep train.log | wc -l)
        echo ${n_prev} previous train.log
        mv ${save_dir}/train.log ${save_dir}/train.log.${n_prev} || true
    fi
    srun -n8 train_ulm.sh --data-dir ${data_dir} --save-dir ${save_dir} --init-method ${init_method}

fi

echo "Run completed at:- "
date



