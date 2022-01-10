#!/bin/bash
#SBATCH -J eval
#SBATCH -o eval_%j.out
#SBATCH -e eval_%j.err
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

export FAIRSEQ_ROOT="/nobackup/users/heting/ssl_disentanglement/fairseq"
export WAV2VEC_DIR="${FAIRSEQ_ROOT}/examples/wav2vec"
export HUBERT_DIR="${FAIRSEQ_ROOT}/examples/hubert"
export ZEROSPEECH_DIR="/nobackup/users/heting/dataset/zerospeech2021_dataset"

MODE=$1
# MODE="EVAL_ABX"
if [ ${MODE} == "ABX" ]; then
    echo "Evaluating ABX"
    # eval model ABX
    pids=()
    layers=(6 12)
    vocab_sizes=(50 100 200)
    
    # model_name="hubert"
    # ckpt_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
    # FEATURE_SCRIPT="$(pwd)/evaluation/abx/discrete_feature.py"

    model_name="hubert_v01_1"
    ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
    FEATURE_SCRIPT="$(pwd)/evaluation/abx/discrete_feature_spk.py"

    km_dataset="librispeech100"
    ext=".pt"
    feature_size=$(bc <<< "scale=2; 1.0/50.0") # 50Hz for hubert
    # splits="dev-clean test-clean dev-other test-other"
    splits="test-clean test-other"

    
    for layer in ${layers[@]}; do
        feat_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}"
        for vocab_size in ${vocab_sizes[@]}; do
            for split in ${splits}; do
            
            data_path="/nobackup/users/heting/dataset/LibriSpeech/"
            abx_item_file="$(pwd)/ABX_data/${split}.item"
            # km_path="$(pwd)/feats/${km_dataset}_l${layer}/km${vocab_size}"
            km_path="${feat_dir}/km${vocab_size}"
            
            output_path="$(pwd)/feats/${model_name}_l${layer}_v${vocab_size}_abx/${split}"
            rm -rf ${output_path}
            mkdir -p ${output_path}
            # if [ -f "${output_path}/ABX_scores.json" ]; then
            #     echo "Skip: Extract discrete feature for layer ${layer} of ${model_name} with vocab size ${vocab_size} using KMeans and save to ${output_path}... "
            #     continue
            # fi
            (
                echo "Extract discrete feature for layer ${layer} of ${model_name} with vocab size ${vocab_size} using KMeans and save to ${output_path}... "
                srun --ntasks=1 --exclusive --gres=gpu:1 --mem=1T -c 16 \
                    python -u "${FEATURE_SCRIPT}" \
                        --layer ${layer} --abx-item-file ${abx_item_file} --km-path ${km_path} \
                        --ckpt-path ${ckpt_path} --data-path ${data_path} --split ${split} --output-path ${output_path} && \ 
                    python -u $(pwd)/libri-light/eval/eval_ABX.py ${output_path} ${abx_item_file} \
                        --file_extension ${ext} --out ${output_path} --feature_size ${feature_size} --cuda
            ) &
            pids+=($!)
            done
        done
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi


if [ ${MODE} == "STOPWORD_EXTRACT" ]; then
    ###########################################################################################################################
    # Extract feature and dump kmean labels for train, dev and test
    ckpt_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
    model_name="hubert"
    FEATURE_SCRIPT="${HUBERT_DIR}/simple_kmeans/dump_hubert_feature.py"

    # ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
    # model_name="hubert_v01_1"
    # FEATURE_SCRIPT="${HUBERT_DIR}/simple_kmeans/dump_hubert_feature_spk.py"

    nshard=1
    # rank=0
    ranks=$(seq 0 $((nshard - 1)))
    eval_datasets="zerospeech_lexical zerospeech_syntactic"

    vocab_sizes="50 100 200"
    km_dataset="librispeech100"
    splits="dev test"

    layer=12

    pids=()
    for eval_dataset in ${eval_datasets[@]}; do
        tsv_dir="${HUBERT_DIR}/manifest/${eval_dataset}"
        label_dir_novocab="${HUBERT_DIR}/manifest/${eval_dataset}/${model_name}_l${layer}_v"
        
        km_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}"
        feat_dir="$(pwd)/feats/${model_name}_${eval_dataset}_l${layer}"
        for split in ${splits[@]}; do
            echo "Extract feature for ${eval_dataset} ${split}..."
            for rank in ${ranks[@]}; do
            # (
                # all_label_exist="True"
                # for vocab_size in ${vocab_sizes[@]}; do
                #     label_dir="${label_dir_novocab}${vocab_size}"
                #     label_path="${label_dir}/${split}_${rank}_${nshard}.km"
                #     if [[ ! -f "${label_path}" ]]; then
                #         echo "${label_path} has not been computed. Compute it."
                #         all_label_exist="False"
                #         break
                #     fi
                # done

                # if [ ${all_label_exist} == "True" ]; then
                #     echo "all labels of ${split} has been computed. Skip!"            
                #     continue
                # fi
                all_label_exist="True"
                for vocab_size in ${vocab_sizes[@]}; do
                    label_path="${label_dir_novocab}${vocab_size}/${split}_${rank}_${nshard}.km"
                    if [[ ! -f "${label_path}" ]]; then
                        echo "${label_path} is missing."
                        all_label_exist="False"
                        break
                    fi
                done
                if [ ${all_label_exist} == "False" ]; then
                    (
                        srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 ./simple_kmeans/extract_hubert_feature.sh \
                            --stage 30 --stop-stage 30 --feature-script ${FEATURE_SCRIPT} \
                            --model-name ${model_name} --split ${split} --layer ${layer} --nshard ${nshard} --rank ${rank} \
                            --ckpt-path ${ckpt_path} --feat-dir ${feat_dir} --tsv-dir ${tsv_dir} \
                            --km-dir ${km_dir} --vocab_sizes "${vocab_sizes}" --label-dir-novocab ${label_dir_novocab} # && \
                            # rm "${feat_dir}/${split}_${rank}_${nshard}.npy" && \
                            # rm "${feat_dir}/${split}_${rank}_${nshard}.len"
                    ) &
                    pids+=($!)
                fi
                # # label output dir
                # echo """
                # srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 1 ./simple_kmeans/extract_hubert_feature.sh \
                #     --stage 30 --stop-stage 30 --feature-script ${FEATURE_SCRIPT} \
                #     --model-name ${model_name} --split ${split} --layer ${layer} --nshard ${nshard} --rank ${rank} \
                #     --ckpt-path ${ckpt_path} --feat-dir ${feat_dir} --tsv-dir ${tsv_dir} \
                #     --km-dir ${km_dir} --vocab_sizes "${vocab_sizes}" --label-dir-novocab ${label_dir_novocab}
                # """
                
                # srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 5 ./simple_kmeans/extract_hubert_feature.sh \
                #     --stage 30 --stop-stage 30 --feature-script ${FEATURE_SCRIPT} \
                #     --model-name ${model_name} --split ${split} --layer ${layer} --nshard ${nshard} --rank ${rank} \
                #     --ckpt-path ${ckpt_path} --feat-dir ${feat_dir} --tsv-dir ${tsv_dir} \
                #     --km-dir ${km_dir} --vocab_sizes "${vocab_sizes}" --label-dir-novocab ${label_dir_novocab}
                
            # ) &
            # pids+=($!)
            done
        done
    done

    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for eval_dataset in ${eval_datasets[@]}; do
        label_dir_novocab="${HUBERT_DIR}/manifest/${eval_dataset}/${model_name}_l${layer}_v"
        for split in ${splits[@]}; do
            for vocab_size in ${vocab_sizes[@]}; do
                label_dir="${label_dir_novocab}${vocab_size}"
                # cp ${label_dir}/${split}_${rank}_${nshard}.km ${label_dir}/${split}.km || true
                for rank in ${ranks}; do
                    if [[ ! -f ${label_dir}/${split}_${rank}_${nshard}.txt ]]; then
                        python simple_kmeans/remove_repeat.py --input-path ${label_dir}/${split}_${rank}_${nshard}.km --output-path ${label_dir}/${split}_${rank}_${nshard}.txt
                    fi
                done
                for rank in ${ranks}; do
                    cat "${label_dir}/${split}_${rank}_${nshard}.km"
                done > "${label_dir}/${split}.km"
                for rank in ${ranks}; do
                    cat "${label_dir}/${split}_${rank}_${nshard}.txt"
                done > "${label_dir}/${split}.txt"
                # cp ${label_dir}/${split}_${rank}_${nshard}.txt ${label_dir}/${split}.txt || true
            done
        done
    done

fi

if [ ${MODE} == "STOPWORD_SCORE" ]; then
    ckpt_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
    model_name="hubert"
    FEATURE_SCRIPT="${HUBERT_DIR}/simple_kmeans/dump_hubert_feature.py"

    # ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
    # model_name="hubert_v01_1"
    # FEATURE_SCRIPT="${HUBERT_DIR}/simple_kmeans/dump_hubert_feature_spk.py"

    eval_datasets="zerospeech_lexical zerospeech_syntactic"
    feat_train_dataset="librilight6k_chunk"

    vocab_sizes="50 100 200"
    km_dataset="librispeech100"
    splits="dev"

    layers=(6 12)

    pids=()

    # for eval_dataset in ${eval_datasets[@]}; do
    #     for layer in ${layers[@]}; do
    #         tsv_dir="${HUBERT_DIR}/manifest/${eval_dataset}"
    #         label_dir_novocab="${HUBERT_DIR}/manifest/${eval_dataset}/${model_name}_l${layer}_v"
    #         for vocab_size in ${vocab_sizes[@]}; do
    #             label_dir="${label_dir_novocab}${vocab_size}"
    #             for split in ${splits[@]}; do
    #             (
    #                 model_path_regex=${HUBERT_DIR}/outputs/ulm_${model_name}_${feat_train_dataset}_l${layer}_v${vocab_size}_8p/checkpoint.best_loss_*.pt
    #                 n_matched=$(ls ${model_path_regex} 2> /dev/null | wc -l)
    #                 if [ ${n_matched} != "1" ]; then
    #                     echo "${n_matched} checkpoints mached. Exit!"
    #                     exit 1
    #                 fi
    #                 model_path=$(ls ${model_path_regex}) # assume only one match

    #                 tsv_path="${tsv_dir}/${split}.tsv"
    #                 label_path="${label_dir_novocab}${vocab_size}/${split}.txt"

    #                 output_dir="${HUBERT_DIR}/feats/${model_name}_l${layer}_v${vocab_size}_score/${eval_dataset//zerospeech_/}"
    #                 # rm -rf ${output_dir} || true
    #                 mkdir -p ${output_dir}
    #                 output_path="${output_dir}/${split}.txt"

    #                 srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 1 python -u ${HUBERT_DIR}/evaluation/lexical_syntactic/score_lm.py \
    #                     --lm ${model_path} --tsv ${tsv_path} --txt ${label_path} --output ${output_path} --batch-size 1000 
                    
    #             ) &
    #             pids+=($!)
    #             done
    #         done
    #     done
    # done
    # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    pids=()
    for layer in ${layers[@]}; do
        for vocab_size in ${vocab_sizes[@]}; do
        (
            output_dir="${HUBERT_DIR}/feats/${model_name}_l${layer}_v${vocab_size}_score"
            srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 zerospeech2021-evaluate \
            ${ZEROSPEECH_DIR} ${output_dir} --no-phonetic --no-semantic -o ${output_dir}
        ) &
        pids+=($!)
        done
    done

    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
echo "Run completed at:- "
date

