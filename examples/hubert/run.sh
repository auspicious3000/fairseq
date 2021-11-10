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

