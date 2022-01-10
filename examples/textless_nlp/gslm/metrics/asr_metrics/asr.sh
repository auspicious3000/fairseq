#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

output_downsample_cut=
manifest_dir=
wav2vec_path=
results_dir=
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

mkdir -p ${manifest_dir}

echo """
*****************************************************************************************************************
    python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
        $output_downsample_cut --valid-percent 0.0  --dest $manifest_dir --ext wav
"""
srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
    $output_downsample_cut --valid-percent 0.0  --dest $manifest_dir --ext wav

echo """
*****************************************************************************************************************
    python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/dummy_asr_data.py --tsv=$manifest_dir/train.tsv \
        --output-path=$manifest_dir/train.ltr
"""
srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/dummy_asr_data.py --tsv=$manifest_dir/train.tsv \
    --output-path=$manifest_dir/train.ltr

cp ${EVAL_DIR}/manifest/librispeech100/dict.ltr.txt ${manifest_dir}

mkdir -p ${results_dir}

echo """
*****************************************************************************************************************
    python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py  \
        $manifest_dir \
        --task audio_finetuning --nbest 1 --path ${wav2vec_path} \
        --gen-subset=train --results-path ${results_dir} \
        --w2l-decoder kenlm --lm-model ${EVAL_DIR}/models/4-gram.bin \
        --lexicon lexicon_ltr.lst --word-score -1 \
        --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 600000 --remove-bpe letter
"""
srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py  \
    $manifest_dir \
    --task audio_finetuning --nbest 1 --path ${wav2vec_path} \
    --gen-subset=train --results-path ${results_dir} \
    --w2l-decoder kenlm --lm-model ${EVAL_DIR}/models/4-gram.bin \
    --lexicon lexicon_ltr.lst --word-score -1 \
    --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 600000 --remove-bpe letter



# FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
# EVAL_DIR="${FAIRSEQ_ROOT}/examples/textless_nlp"

# model_name="hubert"
# tacotron_train_dataset="ljspeech"
# layer=12
# vocab_size=100
# split=dev
# ckpt="checkpoint_20000"

# OUT_DIR=${EVAL_DIR}/outputs/tacotron_${model_name}_${tacotron_train_dataset}_l${layer}_v${vocab_size}_8p/eval_${split}_${ckpt}
# OUTPUT_DOWNSAMPLE_CUT=${OUT_DIR}_16khz_cut
# MANIFEST_DIR=${OUTPUT_DOWNSAMPLE_CUT}/manifest
# mkdir -p ${MANIFEST_DIR}

# python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py  \
#     $OUTPUT_DOWNSAMPLE_CUT --valid-percent 0.0  --dest $MANIFEST_DIR --ext wav

# python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/metrics/asr_metrics/misc/dummy_asr_data.py --tsv=$MANIFEST_DIR/train.tsv \
#     --output-path=$MANIFEST_DIR/train.ltr

# cp ${EVAL_DIR}/manifest/librispeech100/dict.ltr.txt ${MANIFEST_DIR}

# mkdir -p ${results_dir}
# results_dir=${MANIFEST_DIR}_asr

# # run on local machine

# python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py  \
#     $MANIFEST_DIR \
#     --task audio_finetuning --nbest 1 --path ${EVAL_DIR}/models/wav2vec_small_960h.pt \
#     --gen-subset=train --results-path ${results_dir} \
#     --w2l-decoder kenlm --lm-model ${EVAL_DIR}/models/4-gram.bin \
#     --lexicon lexicon_ltr.lst --word-score -1 \
#     --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 600000 --remove-bpe letter

