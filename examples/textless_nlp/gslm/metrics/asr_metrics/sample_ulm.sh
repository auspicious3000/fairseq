#!/bin/bash
source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

ckpt_path=
prompt_path=
output_path=
prefix_size=-1
data_bin=
label_path=
tsv_path=
temperature=0.7
. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

# FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
# HUBERT_DIR="${FAIRSEQ_ROOT}/examples/hubert"
# EVAL_DIR="${FAIRSEQ_ROOT}/examples/textless_nlp"

if [ ! -f ${prompt_path} ]; then
    echo """
*****************************************************************************************************************
    python ${HUBERT_DIR}/simple_kmeans/create_prompt.py --label-path="${label_path}" --tsv-path="${tsv_path}" --prompt-path="${prompt_path}"
    """
    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${HUBERT_DIR}/simple_kmeans/create_prompt.py --label-path="${label_path}" --tsv-path="${tsv_path}" --prompt-path="${prompt_path}"
fi

samples_per_prompt=10
if [ ${temperature} == "0" ]; then
    samples_per_prompt=1
fi
echo """
*****************************************************************************************************************
    python ${EVAL_DIR}/gslm/ulm/sample.py ${data_bin} \
        --path=${ckpt_path} --task=language_modeling --sampling --temperature=${temperature} \
        --seed=1  --prompts=${prompt_path}  --output=${output_path} --max-len-a=0 --max-len-b=500 \
        --prefix-size=${prefix_size} --batch-size=16 --fp16 --samples-per-prompt=${samples_per_prompt}
"""


srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${EVAL_DIR}/gslm/ulm/sample.py ${data_bin} \
        --path=${ckpt_path} --task=language_modeling --sampling --temperature=${temperature} \
        --seed=1  --prompts=${prompt_path}  --output=${output_path} --max-len-a=0 --max-len-b=500 \
        --prefix-size=${prefix_size} --batch-size=16 --fp16 --samples-per-prompt=${samples_per_prompt}





# feat_train_dataset="librilight6k_chunk"
# feat_dev_dataset="librispeech100" # used for both dev and test
# layer=12
# vocab_size=100
# split=dev
# prefix_size=-1

# ckpt_path="${HUBERT_DIR}/models/hubert/hubert_base_ls960.pt"
# model_name="hubert"
# FEATURE_SCRIPT="${HUBERT_DIR}/simple_kmeans/dump_hubert_feature.py"

# label_dir="${HUBERT_DIR}/manifest/${feat_train_dataset}/${model_name}_l${layer}_v${vocab_size}"
# data_bin="${label_dir}/bin"
# ckpt_path_regex=${HUBERT_DIR}/outputs/ulm_${model_name}_${feat_train_dataset}_l${layer}_v${vocab_size}/checkpoint.best_loss_*.pt
# n_matched=$(ls ${ckpt_path} 2> /dev/null | wc -l)
# if [ ${n_matched} != "1" ]; then
#     echo "${n_matched} checkpoints mached. Exit!"
#     exit 1
# fi
# ckpt_path=$(ls ${ckpt_path_regex}) # assume only one match

# prompt_path="${label_dir}/prompt.${split}.txt"
# if [ ! -f ${prompt_path} ]; then
#     label_path="${label_dir}/${split}.km"
#     tsv_path="${HUBERT_DIR}/manifest/${feat_dev_dataset}/${split}.tsv"
#     python ${HUBERT_DIR}/simple_kmeans/create_prompt.py --label-path="${label_path}" --tsv-path="${tsv_path}" --prompt-path="${prompt_path}"
# fi
# output_dir="${label_dir}/ulm_generate"
# mkdir -p ${output_dir}
# output_path="${output_dir}/samples_ps${prefix_size}.txt"

# python ${EVAL_DIR}/gslm/ulm/sample.py ${data_bin} \
#         --path=${ckpt_path} --task=language_modeling --sampling --temperature=0.7 \
#         --seed=1  --prompts=${prompt_path}  --output=${output_path} --max-len-a=0 --max-len-b=500 \
#         --prefix-size=${prefix_size} --batch-size=16 --fp16 --samples-per-prompt=10

