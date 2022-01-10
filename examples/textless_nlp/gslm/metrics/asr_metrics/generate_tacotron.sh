#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

tts_model_path=
quantized_unit_path=
out_dir=
waveglow_path=
code_dict_path=
prompts_description=
nshard=4
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

output_downsample=${out_dir}_16khz
output_downsample_cut=${out_dir}_16khz_cut
rm -rf $out_dir || true
rm -rf ${output_downsample} || true
# rm -rf ${output_downsample_cut} || true
mkdir -p ${out_dir}

split --numeric-suffixes -n l/${nshard} ${quantized_unit_path} ${quantized_unit_path}.
ranks=$(seq 0 $((${nshard} - 1)))
pids=()
for rank in ${ranks[@]}; do
echo """
*****************************************************************************************************************
    PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
        --tts_model_path $tts_model_path \
        --quantized_unit_path "${quantized_unit_path}.0${rank}" \
        --out_audio_dir $out_dir \
        --waveglow_path  $waveglow_path \
        --code_dict_path $code_dict_path \
        --max_decoder_steps 2000
"""
    
    (
    PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech srun --ntasks=1 --exclusive --gres=gpu:1 --mem=64G -c 16 python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
        --tts_model_path $tts_model_path \
        --quantized_unit_path "${quantized_unit_path}.0${rank}" \
        --out_audio_dir $out_dir \
        --waveglow_path  $waveglow_path \
        --code_dict_path $code_dict_path \
        --max_decoder_steps 2000
    ) &
    pids+=($!)
done 
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
mkdir -p ${output_downsample}
mkdir -p ${output_downsample_cut}

echo """
*****************************************************************************************************************
    python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/unit2speech/convert_to_16k.py $out_dir $output_downsample
"""
srun --ntasks=1 --exclusive --gres=gpu:1 --mem=64G -c 16 python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/unit2speech/convert_to_16k.py $out_dir $output_downsample

echo """
*****************************************************************************************************************
    python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/cut_as.py \
        --samples_dir=$output_downsample --out_dir=${output_downsample_cut} \
        --prompts_description=${prompts_description}
"""
srun --ntasks=1 --exclusive --gres=gpu:1 --mem=64G -c 16 python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/cut_as.py \
    --samples_dir=$output_downsample --out_dir=${output_downsample_cut} \
    --prompts_description=${prompts_description}

rm -rf ${out_dir}
rm -rf ${output_downsample}

# FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
# HUBERT_DIR="${FAIRSEQ_ROOT}/examples/hubert"
# EVAL_DIR="${FAIRSEQ_ROOT}/examples/textless_nlp"

# model_name="hubert"
# feat_train_dataset="librilight6k_chunk"
# feat_dev_dataset="librispeech100"
# tacotron_train_dataset="ljspeech"
# layer=12
# vocab_size=100
# split=dev
# prefix_size=-1

# TACOTRON_DIR="/home/heting/workplace/ssl_disentanglement/fairseq/examples/textless_nlp/gslm/unit2speech/tacotron2"
# KM_LABEL_SCRIPT="$(pwd)/simple_kmeans/dump_km_label_batch.py"


# ckpt="checkpoint_20000"

# label_dir="${HUBERT_DIR}/manifest/${feat_train_dataset}/${model_name}_l${layer}_v${vocab_size}"
# prompt_path="${label_dir}/prompt.${split}.txt"

# FAIRSEQ_ROOT=/nobackup/users/heting/ssl_disentanglement/fairseq
# TTS_MODEL_PATH=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/${ckpt}
# QUANTIZED_UNIT_PATH=${prompt_path}
# OUT_DIR=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${ckpt}
# OUTPUT_DOWNSAMPLE=${OUT_DIR}_16khz
# OUTPUT_DOWNSAMPLE_CUT=${OUT_DIR}_16khz_cut
# WAVEGLOW_PATH=${EVAL_DIR}/models/waveglow_256channels_new.pt
# CODE_DICT_PATH=${EVAL_DIR}/manifest/code_dict_v${vocab_size}
# rm -rf $OUT_DIR || true
# mkdir -p ${OUT_DIR}


# PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
#     --tts_model_path $TTS_MODEL_PATH \
#     --quantized_unit_path $QUANTIZED_UNIT_PATH \
#     --out_audio_dir $OUT_DIR \
#     --waveglow_path  $WAVEGLOW_PATH \
#     --code_dict_path $CODE_DICT_PATH \
#     --max_decoder_steps 2000

# mkdir -p ${OUTPUT_DOWNSAMPLE}
# mkdir -p ${OUTPUT_DOWNSAMPLE_CUT}

# python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/unit2speech/convert_to_16k.py $OUT_DIR $OUTPUT_DOWNSAMPLE

# PROMPTS_DESCRIPTION=${EVAL_DIR}/manifest/${feat_dev_dataset}/ground_truth_continuation_${split}.json
# python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/cut_as.py \
#     --samples_dir=$OUTPUT_DOWNSAMPLE --out_dir=$OUTPUT_DOWNSAMPLE_CUT \
#     --prompts_description=${PROMPTS_DESCRIPTION}