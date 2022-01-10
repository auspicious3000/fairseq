#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

stage=0
stop_stage=100
model_name=
layer=
vocab_size=
split=test
temperature=
tacotron_ckpt="checkpoint_20000"
prefix_size=-1 # prefix to sample ulm
wav2vec=
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

export FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
export HUBERT_DIR="${FAIRSEQ_ROOT}/examples/hubert"
export EVAL_DIR="${FAIRSEQ_ROOT}/examples/textless_nlp"

feat_train_dataset="librilight6k_chunk"
feat_dev_dataset="librispeech100" # used for both dev and test
tacotron_train_dataset="ljspeech"

# Example parameters:
# model_name="hubert_v01_1"
# layer=6
# vocab_size=100
# split=dev
# tacotron_ckpt="checkpoint_20000"
# prefix_size=-1 # prefix to sample ulm
# wav2vec="wav2vec_small_960h.pt"

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: Sample from language model..."
    label_dir="${HUBERT_DIR}/manifest/${feat_train_dataset}/${model_name}_l${layer}_v${vocab_size}"
    label_path="${label_dir}/${split}.km"
    data_bin="${label_dir}/bin"
    prompt_path="${label_dir}/prompt_t${temperature}.${split}.txt"
    tsv_path="${HUBERT_DIR}/manifest/${feat_dev_dataset}/${split}.tsv"
    output_dir="${label_dir}/ulm_generate"
    mkdir -p ${output_dir}
    output_path="${output_dir}/samples_ps${prefix_size}_${split}_t${temperature}.txt"
    
    if [ -f ${output_path} ] && [ "$(wc -l < ${output_path})" -gt 10000 ]; then
        echo "output_path ${output_path} already exist. Skip Stage 10."
    else
        ckpt_path_regex=${HUBERT_DIR}/outputs/ulm_${model_name}_${feat_train_dataset}_l${layer}_v${vocab_size}_8p/checkpoint.best_loss_*.pt
        n_matched=$(ls ${ckpt_path_regex} 2> /dev/null | wc -l)
        if [ ${n_matched} != "1" ]; then
            echo "${n_matched} checkpoints mached. Exit!"
            exit 1
        fi
        ckpt_path=$(ls ${ckpt_path_regex}) # assume only one match

        ${EVAL_DIR}/gslm/metrics/asr_metrics/sample_ulm.sh \
            --ckpt-path ${ckpt_path} \
            --prompt-path ${prompt_path} \
            --output-path ${output_path} \
            --label-path ${label_path} \
            --tsv-path ${tsv_path} \
            --prefix-size ${prefix_size} \
            --data-bin ${data_bin} \
            --temperature ${temperature}
    
    fi
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
    echo "Stage 20: Generate audio using Tacotron..."
    label_dir="${HUBERT_DIR}/manifest/${feat_train_dataset}/${model_name}_l${layer}_v${vocab_size}"
    output_dir="${label_dir}/ulm_generate"
    tts_model_path=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/${tacotron_ckpt}
    quantized_unit_path="${output_dir}/samples_ps${prefix_size}_${split}_t${temperature}.txt"
    out_dir=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${tacotron_ckpt}_t${temperature}
    waveglow_path=${EVAL_DIR}/models/waveglow_256channels_new.pt
    code_dict_path=${EVAL_DIR}/manifest/code_dict_v${vocab_size}
    prompts_description=${EVAL_DIR}/manifest/${feat_dev_dataset}/ground_truth_continuation_${split}.json

    ${EVAL_DIR}/gslm/metrics/asr_metrics/generate_tacotron.sh \
        --tts-model-path ${tts_model_path} \
        --quantized-unit-path ${quantized_unit_path} \
        --out-dir ${out_dir} \
        --waveglow-path ${waveglow_path} \
        --code-dict-path ${code_dict_path} \
        --prompts-description ${prompts_description}

fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ]; then
    echo "Stage 30: Transcribe audio using wav2vec"
    out_dir=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${tacotron_ckpt}_t${temperature}
    output_downsample_cut=${out_dir}_16khz_cut
    manifest_dir=${output_downsample_cut}/manifest
    wav2vec_path=${EVAL_DIR}/models/${wav2vec}
    asr_output=${manifest_dir}_asr
    ${EVAL_DIR}/gslm/metrics/asr_metrics/asr.sh \
        --output-downsample-cut ${output_downsample_cut} \
        --manifest-dir ${manifest_dir} \
        --wav2vec-path ${wav2vec_path} \
        --results-dir ${asr_output}
fi

if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ]; then
    echo "Stage 40: Evaluate using ASR metrics"
    
    out_dir=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${tacotron_ckpt}_t${temperature}
    output_downsample_cut=${out_dir}_16khz_cut
    manifest_dir=${output_downsample_cut}/manifest
    prompts_description=${EVAL_DIR}/manifest/${feat_dev_dataset}/ground_truth_continuation_${split}.json
    asr_output=${manifest_dir}_asr
    asr_transcript=${asr_output}/hypo.word-${wav2vec}-train.txt

    ${EVAL_DIR}/gslm/metrics/asr_metrics/metric.sh \
        --asr-transcript ${asr_transcript} \
        --manifest-dir ${manifest_dir} \
        --prompts-description ${prompts_description}
fi

if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ]; then
    echo "Stage 50: Evaluate using ASR metrics on oracle text"
    gt_manifest_dir=${EVAL_DIR}/manifest/${feat_dev_dataset}
    gt_manifest_tsv=${gt_manifest_dir}/${split}.tsv
    gt_manifest_ltr=${gt_manifest_dir}/${split}.ltr
    
    out_dir=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${tacotron_ckpt}_gt
    manifest_dir=${out_dir}/gt_manifest
    mkdir -p ${manifest_dir}

    gt_manifest_tsv_out=${manifest_dir}/train.tsv
    gt_manifest_ltr_out=${manifest_dir}/gt_${split}.ltr
    cp ${gt_manifest_tsv} ${gt_manifest_tsv_out}

    prompts_description=${EVAL_DIR}/manifest/${feat_dev_dataset}/ground_truth_continuation_${split}.json

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${EVAL_DIR}/gslm/metrics/asr_metrics/convert_ltr.py \
        --transcript ${gt_manifest_ltr} --output ${gt_manifest_ltr_out}

    ${EVAL_DIR}/gslm/metrics/asr_metrics/metric.sh \
        --asr-transcript ${gt_manifest_ltr_out} \
        --manifest-dir ${manifest_dir} \
        --prompts-description ${prompts_description}
fi