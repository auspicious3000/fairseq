#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

asr_transcript=
manifest_dir=
prompts_description=
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail
pids=()

if [ -f ${manifest_dir}/ppx.txt ] && [ "$(awk '!/^(srun:)/' ${manifest_dir}/ppx.txt | wc -l)" -gt 6 ]; then
    echo "${manifest_dir}/ppx.txt has been computed."
else
    (
    echo """*****************************************************************************************************************
    python ${EVAL_DIR}/gslm/metrics/asr_metrics/ppx.py --asr-transcript=${asr_transcript} --cut-tail \
        --manifest=${manifest_dir}/train.tsv --prompts-description=${prompts_description} 2>&1 | tee ${manifest_dir}/ppx.txt
    """
    start_time=$(date +%s)
    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${EVAL_DIR}/gslm/metrics/asr_metrics/ppx.py --asr-transcript=${asr_transcript} --cut-tail \
        --manifest=${manifest_dir}/train.tsv --prompts-description=${prompts_description} 2>&1 | tee ${manifest_dir}/ppx.txt
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "Took ${elapsed}s to finish Computing ${manifest_dir}/ppx.txt."
    )&
    pids+=($!)
fi



if [ -f ${manifest_dir}/self_auto_bleu.txt ] && [ "$(awk '!/^(srun:)/' ${manifest_dir}/self_auto_bleu.txt | wc -l)" -gt 7 ]; then
    echo "${manifest_dir}/self_auto_bleu.txt has been computed."
else
    (
    echo """*****************************************************************************************************************
    python ${EVAL_DIR}/gslm/metrics/asr_metrics/self_auto_bleu.py --asr-transcript=${asr_transcript} --cut-tail \
        --manifest=${manifest_dir}/train.tsv --prompts-description=${prompts_description} 2>&1 | tee ${manifest_dir}/self_auto_bleu.txt
    """
    start_time=$(date +%s)
    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${EVAL_DIR}/gslm/metrics/asr_metrics/self_auto_bleu.py --asr-transcript=${asr_transcript} --cut-tail \
        --manifest=${manifest_dir}/train.tsv --prompts-description=${prompts_description} 2>&1 | tee ${manifest_dir}/self_auto_bleu.txt
    
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "Took ${elapsed}s to finish Computing ${manifest_dir}/self_auto_bleu.txt."
    ) &
    pids+=($!)
fi


i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "metric.sh: ${i} background jobs are failed." && false
# FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
# EVAL_DIR="${FAIRSEQ_ROOT}/examples/textless_nlp"

# feat_dev_dataset="librispeech100"
# model_name="hubert"
# tacotron_train_dataset="ljspeech"
# layer=12
# vocab_size=100
# split=dev
# ckpt="checkpoint_20000"

# WAV2VEC=wav2vec_small_960h.pt

# OUT_DIR=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${ckpt}
# OUTPUT_DOWNSAMPLE_CUT=${OUT_DIR}_16khz_cut
# MANIFEST_DIR=${OUTPUT_DOWNSAMPLE_CUT}/manifest
# ASR_OUTPUT=${MANIFEST_DIR}_asr
# PROMPTS_DESCRIPTION=${EVAL_DIR}/manifest/${feat_dev_dataset}/ground_truth_continuation_${split}.json

# python ${EVAL_DIR}/gslm/metrics/asr_metrics/ppx.py --asr-transcript=$ASR_OUTPUT/hypo.word-${WAV2VEC}-train.txt --cut-tail \
#     --manifest=$MANIFEST_DIR/train.tsv --prompts-description=${PROMPTS_DESCRIPTION} 2>&1 | tee ${MANIFEST_DIR}/ppx.txt


# python ${EVAL_DIR}/gslm/metrics/asr_metrics/self_auto_bleu.py --asr-transcript=$ASR_OUTPUT/hypo.word-${WAV2VEC}-train.txt --cut-tail \
#     --manifest=$MANIFEST_DIR/train.tsv --prompts-description=${PROMPTS_DESCRIPTION} 2>&1 | tee ${MANIFEST_DIR}/self_auto_bleu.txt


# python ${EVAL_DIR}/gslm/metrics/asr_metrics/continuation_eval.py --asr-transcript=$ASR_OUTPUT/hypo.word-${WAV2VEC}-train.txt \
#     --manifest=$MANIFEST_DIR/train.tsv --prompts-description=${PROMPTS_DESCRIPTION} 2>&1 | tee ${MANIFEST_DIR}/continuation_eval.txt