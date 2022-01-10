#!/bin/bash
#SBATCH -J eval_generate
#SBATCH -o eval_generate_%j.out
#SBATCH -e eval_generate_%j.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

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

set -e

export FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"

tacotron_ckpt="checkpoint_20000"
prefix_size=-1 # prefix to sample ulm
wav2vec="wav2vec_big_960h.pt"

model_names=("hubert") # missing 50
layers=(6 12)

# model_names=("hubert_v01_1")
# layers=(12)

# vocab_sizes=(50 100 200)
vocab_sizes=(50 200)
splits=("test")
# temperatures=(0)
# temperatures=(0.3 0.5 0.7 1 1.2 1.5 2.1 3)
temperatures=(0.5 1.2 2.1 3)


pids=()
for model_name in "${model_names[@]}"; do
    for layer in "${layers[@]}"; do
        for vocab_size in "${vocab_sizes[@]}"; do
            for split in "${splits[@]}"; do
                for temperature in ${temperatures[@]}; do
                    echo "Evaluating model_name (${model_name}) layer (${layer}) vocab_size (${vocab_size}) split (${split}) temperature (${temperature})"
                    (
                        # srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 1 
                        ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/metrics/asr_metrics/eval.sh \
                            --stage 0 --stop-stage 40 \
                            --model-name ${model_name} \
                            --layer ${layer} \
                            --vocab-size ${vocab_size} \
                            --split $split \
                            --tacotron-ckpt ${tacotron_ckpt} \
                            --prefix-size ${prefix_size} \
                            --wav2vec ${wav2vec} \
                            --temperature ${temperature}
                    ) &
                    pids+=($!)
                done
            done
        done
    done
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "eval_generate: ${i} background jobs are failed."


echo "Run completed at:- "
date