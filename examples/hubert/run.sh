#!/bin/bash

source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

stage=10
stop_stage=10
data_dir="/nobackup/users/kzqian/data/hubert/librispeech"

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 1: Training hubert"
    manifest_dir="$(pwd)/manifest/librispeech/meta" 
    label_dir="$(pwd)/manifest/librispeech/label_100_7"
    spk2info="$(pwd)/manifest/librispeech/spk2info.dict"
    python $(pwd)/../../fairseq_cli/hydra_train.py \
        --config-dir $(pwd)/config/ssl_disentangle \
        --config-name hubert_base_librispeech \
        task.data=${manifest_dir} task.label_dir=${label_dir} task.spk2info=${spk2info} task.labels='["km"]' model.label_rate=50
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "Stage 1: Training hubertckpt1"
    manifest_dir="$(pwd)/manifest/librispeech/meta"
    label_dir="$(pwd)/manifest/librispeech/label_100_7"
    spk2info="$(pwd)/manifest/librispeech/spk2info.dict"
    hubert_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
    export SLURM_NNODES=1
    export SLURM_NTASKS_PER_NODE=1
    python $(pwd)/../../fairseq_cli/hydra_train.py \
        --config-dir $(pwd)/config/ssl_disentangle \
        --config-name hubertckpt1_base_librispeech \
        task.data=${manifest_dir} task.label_dir=${label_dir} task.spk2info=${spk2info} task.labels='["km"]' \
        model.label_rate=50 model.pretrained_hubert_path=${hubert_path}
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "Stage 1: Training hubert"
    manifest_dir="$(pwd)/manifest/librispeech/meta"
    label_dir="$(pwd)/manifest/librispeech/label_100_7"
    spk2info="$(pwd)/manifest/librispeech/spk2info.dict"
    expdir="$(pwd)/outputs/v04"
    python $(pwd)/../../fairseq_cli/hydra_train.py \
        --config-dir $(pwd)/config/ssl_disentangle \
        --config-name hubert4_librispeech \
	hydra.run.dir=${expdir} \
        common.log_file=train.log \
        task.data=${manifest_dir} task.label_dir=${label_dir} task.spk2info=${spk2info} task.labels='["km_np"]' model.label_rate=50
fi
