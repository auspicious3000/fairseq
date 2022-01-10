#!/bin/bash
#SBATCH -J extracthubertfeature
#SBATCH -o extracthubertfeature_%j.out
#SBATCH -e extracthubertfeature_%j.err
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

set -e
PARALLEL="True"

# Model Dependent Setting 

# ckpt_path="$(pwd)/models/hubert/hubert_base_ls960.pt"
# model_name="hubert"
# FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature.py"

ckpt_path="/nobackup/users/kzqian/ssl-disentangle/baseline/checkpoints/checkpoint_best.pt"
model_name="hubert_base"
FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature.py"

# ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
# model_name="hubert_v01_1"
# FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature_spk.py"

# Model Independent Setting
KM_LABEL_SCRIPT="$(pwd)/simple_kmeans/dump_km_label_batch.py"
train_dataset="librilight6k_chunk"
km_dataset="librispeech100"

# ULM_VOCAB_SIZE=50
vocab_sizes="50 100 200"
layer=8

if [ 0 -eq 0 ]; then
    
    tsv_dir="$(pwd)/manifest/${km_dataset}"
    feat_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}" # do not differentiate by vocab_size

    mkdir -p ${feat_dir}

    pids=()
    
    ###########################################################################################################################
    # Extract feature and learn keam for train
    split="train"
    nshard=4
    ranks=$(seq 0 $((nshard - 1)))

    echo "Extract feature for ${km_dataset} ${split}..."

    all_km_exist="True"
    for vocab_size in ${vocab_sizes[@]}; do
        km_path="${feat_dir}/km${vocab_size}"
        if [[ ! -f "${km_path}" ]]; then
            echo "${km_path} is missing."
            all_km_exist="False"
            break
        fi
    done

    if [ ${all_km_exist} == "False" ]; then
        echo "Missing km. Extract features"
        for rank in ${ranks[@]}; do
            if [[ -f "${feat_dir}/${split}_${rank}_${nshard}.npy" && -f "${feat_dir}/${split}_${rank}_${nshard}.len"  ]]; then
                echo "Feature Extraction: rank ${rank} / ${nshard} has been computed. Skip."
                continue
            fi
            (
                echo "Parallel ${rank}"
                srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${FEATURE_SCRIPT} \
                    "${tsv_dir}" "${split}" \
                    "${ckpt_path}" "${layer}" "${nshard}" "${rank}" "${feat_dir}"
            )&
            pids+=($!)
        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    fi

    ###########################################################################################################################
    # Compute KMeans using features from train split
    echo "Compute KMeans using ${km_dataset} ${split}"
    pids=()
    for vocab_size in ${vocab_sizes[@]}; do
        km_path="${feat_dir}/km${vocab_size}"
        if [[ -f "${km_path}" ]]; then
            echo "${km_path} has been computed. Skip."
            continue
        fi
        (
            
            echo "Params: split (${split}) vocab_size (${vocab_size})"
            echo "Params: km_path (${km_path})"
            echo "Params: feat_dir (${feat_dir})"
            srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python simple_kmeans/learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${vocab_size} --percent -1
        ) &
        pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for rank in ${ranks[@]}; do
        rm "${feat_dir}/${split}_${rank}_${nshard}.npy" || true
        rm "${feat_dir}/${split}_${rank}_${nshard}.len" || true
    done 

    ###########################################################################################################################
    # Extract feature and dump kmean labels for dev and test
    splits=(dev test)
    nshard=1
    rank=0
    for split in ${splits[@]}; do
        echo "Extract feature for ${split}..."
        srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${FEATURE_SCRIPT} \
                "${tsv_dir}" "${split}" \
                "${ckpt_path}" "${layer}" "${nshard}" "${rank}" "${feat_dir}"

        for vocab_size in ${vocab_sizes[@]}; do
            echo "Dump label for ${split} with vocab size of ${vocab_size}..."
            km_path="${feat_dir}/km${vocab_size}"
            label_dir="$(pwd)/manifest/${train_dataset}/${model_name}_l${layer}_v${vocab_size}"
            mkdir -p ${label_dir}
            srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${KM_LABEL_SCRIPT} \
                "${feat_dir}" "${split}" "${km_path}" "${nshard}" "${rank}" "${label_dir}" --disable_tqdm False

            mv ${label_dir}/${split}_${rank}_${nshard}.km ${label_dir}/${split}.km

            python simple_kmeans/remove_repeat.py --input-path ${label_dir}/${split}.km --output-path ${label_dir}/${split}.txt
        done
    done
fi


if [ 0 -eq 0 ]; then
    train_pids=()
    # for vocab_size in ${vocab_sizes[@]}; do
    split="train"
    nshard=160
    ranks=$(seq 0 $((nshard - 1)))
    tsv_dir="$(pwd)/manifest/${train_dataset}"
    # differentiate using vocab_size so that multiple jobs can be run parallelly on slurm
    feat_dir="$(pwd)/feats/${model_name}_${train_dataset}_l${layer}"
    
    rm -rf ${feat_dir} || true
    mkdir -p ${feat_dir}

    pids=()
    
    label_dir_novocab="$(pwd)/manifest/${train_dataset}/${model_name}_l${layer}_v"
    # km_path="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}/km${vocab_size}"
    km_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}"

    ###########################################################################################################################
    # Extract features for train_dataset
    echo "Extract feature for ${split}..."
    

    # label output dir
    for rank in ${ranks[@]}; do
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
                    --km-dir ${km_dir} --vocab_sizes "${vocab_sizes}" --label-dir-novocab ${label_dir_novocab} && \
                    rm "${feat_dir}/${split}_${rank}_${nshard}.npy" && \
                    rm "${feat_dir}/${split}_${rank}_${nshard}.len"
            ) &
            pids+=($!)
        fi
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Finished Extracting features for ${split}."

    rm -rf ${feat_dir} || true

    
    for vocab_size in ${vocab_sizes[@]}; do
        
        label_dir=${label_dir_novocab}${vocab_size}
        if [ ! -f ${label_dir}/${split}.km ]; then
            for rank in ${ranks}; do
                cat "${label_dir}/${split}_${rank}_${nshard}.km"
            done > "${label_dir}/${split}.km"
        fi

        if [ ! -f ${label_dir}/${split}.txt ]; then
            python simple_kmeans/remove_repeat.py --input-path ${label_dir}/${split}.km --output-path ${label_dir}/${split}.txt
        fi
    done

    for vocab_size in ${vocab_sizes[@]}; do
        ###########################################################################################################################
        # Prepare data and Train
        tsv_dir="$(pwd)/manifest/${train_dataset}"
        label_dir="$(pwd)/manifest/${train_dataset}/${model_name}_l${layer}_v${vocab_size}"
        data_dir="${label_dir}/bin"
        all_data_exist="True"
        all_files=(dict.txt test.bin test.idx train.bin train.idx valid.bin valid.idx)
        for file in ${all_files[@]}; do
            if [ ! -f ${data_dir}/${file} ]; then
                all_data_exist="False"
                break
            fi
        done
        if [ ${all_data_exist} == "False" ]; then
            rm -rf ${data_dir}
            
            echo "Params: tsv_dir   (${tsv_dir})"
            echo "Params: label_dir (${label_dir})"
            echo "Params: data_dir  (${data_dir})"
            
            ###########################################################################################################################
            ## echo "Prepare data ..."
            srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 fairseq-preprocess --only-source \
                --trainpref ${label_dir}/train.txt --validpref ${label_dir}/dev.txt --testpref ${label_dir}/test.txt \
                --destdir ${data_dir} --workers 5
        else
            echo "All data exist in ${data_dir}" 
        fi
        echo "Train ULM..."

        ###########################################################################################################################
        ## echo "Train the model ..."
        # save_dir="$(pwd)/outputs/ulm_${model_name}_${train_dataset}_l${layer}_v${vocab_size}_8p"
        
        # echo "Params: save_dir  (${save_dir})"
        # mkdir -p ${save_dir}
        # if [[ -f "${save_dir}/train.log" ]]; then
        #     n_prev=$(ls -1q ${save_dir} | grep train.log | wc -l)
        #     echo ${n_prev} previous train.log
        #     mv ${save_dir}/train.log ${save_dir}/train.log.${n_prev} || true
        # fi
        # (
        #     srun --ntasks=1 --gres=gpu:1 --mem=128G fairseq-train ${data_dir} \
        #         --task=language_modeling \
        #         --arch=transformer_lm_big \
        #         --share-decoder-input-output-embed \
        #         --dropout=0.1 \
        #         --attention-dropout=0.1 \
        #         --optimizer=adam \
        #         --adam-betas='(0.9, 0.98)' \
        #         --clip-norm=1.0 \
        #         --lr=0.0005 \
        #         --lr-scheduler=inverse_sqrt \
        #         --warmup-updates=4000 \
        #         --warmup-init-lr=1e-07 \
        #         --tokens-per-sample=3072 \
        #         --update-freq=16 \
        #         --max-tokens=4096 \
        #         --num-workers=4 \
        #         --skip-invalid-size-inputs-valid-test \
        #         --max-update=500000 \
        #         --log-interval=10 \
        #         --seed=100502 \
        #         --fp16 \
        #         --sample-break-mode=eos \
        #         --keep-best-checkpoints=1 \
        #         --save-dir=${save_dir} \
        #         --log-file=train.log \
        #         --no-epoch-checkpoints \
        #         --keep-last-epochs=1 \
        #         --keep-interval-updates=1 \
        #         --save-interval-updates=500 \
        #         --save-interval=1000 \
        #         --ddp-backend=no_c10d \
        #         --distributed-world-size=2 \
        #         --distributed-rank=${rank} \
        #         2>&1 | tee ${save_dir}/train.log
        # ) &
        # train_pids+=($!)
    done

    i=0; for train_pids in "${train_pids[@]}"; do wait ${train_pids} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


fi

echo "Run completed at:- "
date



