#!/bin/bash


expdir=$(pwd)/outputs/v01
mkdir -p $expdir

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE_JOB=\$SLURM_NTASKS
RANK_NODE=\$SLURM_NODEID
PROC_PER_NODE=4
MASTER_ADDR_JOB=\$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"
DDP_BACKEND=c10d

manifest_dir="$(pwd)/manifest/librispeech/meta"
label_dir="$(pwd)/manifest/librispeech/label_100_7"
spk2info="$(pwd)/manifest/librispeech/spk2info.dict"
hubert_path="$(pwd)/models/hubert/hubert_base_ls960.pt"

# 1st iteration HuBERT pre-training (100k steps)
HYDRA_FULL_ERROR=1 python -u $(pwd)/../../fairseq_cli/hydra_train.py  \
    --config-dir $(pwd)/config/ssl_disentangle \
    --config-name hubert_base_librispeech \
    hydra.run.dir=${expdir} \
    common.log_file=train.log \
    task.data=${manifest_dir} \
    task.label_dir=${label_dir} \
    task.labels=["km"] \
    task.spk2info=${spk2info} \
    dataset.train_subset=train \
    dataset.valid_subset=valid \
    model.label_rate=50 \
    optimization.update_freq=[2] \
    optimization.max_update=100000 \
    lr_scheduler.warmup_updates=8000 \
    2>&1 | tee ${expdir}/train.log

# set distributed_training.distributed_port=0 for srun

