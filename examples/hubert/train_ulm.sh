#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

data_dir=
save_dir=
init_method=
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

NUM_GPU_PER_NODE=4
rank=${SLURM_PROCID}
device_id=$((${rank} % ${NUM_GPU_PER_NODE}))

echo "rank: ${rank} device id: ${device_id}"
fairseq-train ${data_dir} \
    --task=language_modeling \
    --arch=transformer_lm_big \
    --share-decoder-input-output-embed \
    --dropout=0.1 \
    --attention-dropout=0.1 \
    --optimizer=adam \
    --adam-betas='(0.9, 0.98)' \
    --clip-norm=1.0 \
    --lr=0.0005 \
    --lr-scheduler=inverse_sqrt \
    --warmup-updates=4000 \
    --warmup-init-lr=1e-07 \
    --tokens-per-sample=3072 \
    --update-freq=16 \
    --max-tokens=4096 \
    --num-workers=4 \
    --skip-invalid-size-inputs-valid-test \
    --max-update=500000 \
    --log-interval=10 \
    --seed=100502 \
    --fp16 \
    --sample-break-mode=eos \
    --keep-best-checkpoints=1 \
    --save-dir=${save_dir} \
    --log-file=train.log \
    --no-epoch-checkpoints \
    --keep-last-epochs=1 \
    --keep-interval-updates=1 \
    --save-interval-updates=500 \
    --save-interval=1000 \
    --ddp-backend=no_c10d \
    --distributed-world-size=8 \
    --distributed-rank=${rank} \
    --distributed-num-procs=1 \
    --distributed-init-method=${init_method} \
    --distributed-no-spawn \
    --device-id=${device_id}\
    2>&1 | tee ${save_dir}/train.log