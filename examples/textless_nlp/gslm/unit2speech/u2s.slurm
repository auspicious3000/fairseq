#!/bin/bash
#SBATCH -J u2s
#SBATCH -o u2s_%j.out
#SBATCH -e u2s_%j.err
#SBATCH --mail-user=heting@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=5
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

# ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v01_1/checkpoints/checkpoint_random.pt"
# model_name="hubert_v01_1"
# FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature_spk.py"

ckpt_path="/nobackup/users/kzqian/ssl-disentangle/baseline/checkpoints/checkpoint_best.pt"
model_name="hubert_base"
FEATURE_SCRIPT="$(pwd)/simple_kmeans/dump_hubert_feature.py"

# Model Independent Setting
TACOTRON_DIR="/home/heting/workplace/ssl_disentanglement/fairseq/examples/textless_nlp/gslm/unit2speech/tacotron2"
KM_LABEL_SCRIPT="$(pwd)/simple_kmeans/dump_km_label_batch.py"
train_dataset="ljspeech"
km_dataset="librispeech100"

# vocab_sizes=(50) # 151282
# vocab_sizes=(100) # 151284
# vocab_sizes=(200) # 151693
layer=8

if [ 0 -eq 0 ]; then
    conda activate $PYTHON_VIRTUAL_ENVIRONMENT
    tsv_dir="$(pwd)/manifest/${km_dataset}"
    feat_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}" # do not differentiate by vocab_size
    vocab_sizes=(50 100 200)
    mkdir -p ${feat_dir}

    pids=()
    
    ###########################################################################################################################
    # Extract feature and learn kmean for train
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
                srun --ntasks=1 --exclusive --gres=gpu:1 --mem=64G -c 1 python ${FEATURE_SCRIPT} \
                    "${tsv_dir}" "${split}" \
                    "${ckpt_path}" "${layer}" "${nshard}" "${rank}" "${feat_dir}"
            )&
            pids+=($!)
        done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    else
        echo "km has been found. Do not extract features"
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
            srun --ntasks=1 --exclusive --gres=gpu:1 --mem=128G -c 1 python simple_kmeans/learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${vocab_size} --percent -1
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
    # Extract feature and dump kmean labels for train, dev and test
    splits=(train dev test)
    nshard=1
    rank=0
    tsv_dir="$(pwd)/manifest/${train_dataset}_16khz"
    vocab_sizes="50 100 200"
    label_dir_novocab="$(pwd)/manifest/${train_dataset}/${model_name}_l${layer}_v"

    for split in ${splits[@]}; do
        all_label_exist="True"
        for vocab_size in ${vocab_sizes[@]}; do
            label_dir="${label_dir_novocab}${vocab_size}"
            label_path="${label_dir}/${split}_${rank}_${nshard}.km"
            if [[ ! -f "${label_path}" ]]; then
                echo "${label_path} has not been computed. Skip."
                all_label_exist="False"
                break
            fi
        done

        if [ ${all_label_exist} == "True" ]; then
            echo "all labels of ${split} has been computed. Skip!"            
            continue
        fi
        echo "Extract feature for ${split}..."
        km_dir="$(pwd)/feats/${model_name}_${km_dataset}_l${layer}"
        feat_dir="$(pwd)/feats/${model_name}_${train_dataset}_l${layer}"
         # label output dir
        srun --ntasks=1 --exclusive --gres=gpu:1 --mem=64G -c 1 ./simple_kmeans/extract_hubert_feature.sh \
            --stage 30 --stop-stage 30 --feature-script ${FEATURE_SCRIPT} \
            --model-name ${model_name} --split ${split} --layer ${layer} --nshard ${nshard} --rank ${rank} \
            --ckpt-path ${ckpt_path} --feat-dir ${feat_dir} --tsv-dir ${tsv_dir} \
            --km-dir ${km_dir} --vocab_sizes "${vocab_sizes}" --label-dir-novocab ${label_dir_novocab}
    done

    for split in ${splits[@]}; do
        for vocab_size in ${vocab_sizes[@]}; do
            label_dir="${label_dir_novocab}${vocab_size}"
            cp ${label_dir}/${split}_${rank}_${nshard}.km ${label_dir}/${split}.km || true
            if [[ ! -f ${label_dir}/${split}_${rank}_${nshard}.txt ]]; then
                python simple_kmeans/remove_repeat.py --input-path ${label_dir}/${split}_${rank}_${nshard}.km --output-path ${label_dir}/${split}_${rank}_${nshard}.txt
            fi
            cp ${label_dir}/${split}_${rank}_${nshard}.txt ${label_dir}/${split}.txt || true
        done
    done
fi

if [ 0 -eq 0 ]; then
    conda activate ssl # tacotron environment
    vocab_sizes=(100)
    pids=()
    current_dir="$(pwd)"
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    export MASTER_PORT=12342
    export WORLD_SIZE=8
    # export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG_SUBSYS=ALL

    echo "vocab_size (${vocab_sizes})"

    for vocab_size in ${vocab_sizes[@]}; do
        (
            echo "train on ${layer}-layer ${model_name} features of ${train_dataset} with vocab size: ${vocab_size}..."
            train_label_param="training_labels=${current_dir}/manifest/${train_dataset}/${model_name}_l${layer}_v${vocab_size}/train.km"
            train_audio_param="training_audiopaths=${current_dir}/manifest/${train_dataset}/train.tsv"
            valid_label_param="validation_labels=${current_dir}/manifest/${train_dataset}/${model_name}_l${layer}_v${vocab_size}/dev.km"
            valid_audio_param="validation_audiopaths=${current_dir}/manifest/${train_dataset}/dev.tsv"
            save_dir="${current_dir}/outputs/tacotron_${model_name}_${train_dataset}_l${layer}_v${vocab_size}_8p"
            mkdir -p ${save_dir}
            if [[ -f "${save_dir}/train.log" ]]; then
                n_prev=$(ls -1q ${save_dir} | grep train.log | wc -l)
                echo "${n_prev} previous train.log"
                mv ${save_dir}/train.log ${save_dir}/train_${n_prev}.log || true
            fi
            pretrained_tacotron="${TACOTRON_DIR}/models/tacotron2_statedict.pt"
            n_symbols="$((${vocab_size}+2))"

            cd ${TACOTRON_DIR}/..

            OMP_NUM_THREADS=1 srun python -u train.py -o "${save_dir}" -l "${save_dir}" -c ${pretrained_tacotron} --warm_start \
            --hparams="${train_label_param},${train_audio_param},${valid_label_param},${valid_audio_param},
            feature_size=0.02,vocab_size=${vocab_size},dataloader=TextIntMelLoader,base_chunk_size=50,extra_chunk_size_per_epoch=5,n_symbols=${n_symbols}" 2>&1 | tee ${save_dir}/train.log
            cd ${current_dir}
        ) &
        pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

echo "Run completed at:- "
date
