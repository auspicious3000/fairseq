#!/bin/bash

#./prepare_label.sh 4 7 /nobackup/users/kzqian/data/hubert/librispeech/meta_org /nobackup/users/kzqian/data/hubert/librispeech/feat_perm /nobackup/users/kzqian/data/hubert/librispeech/label_perm_100_7 /nobackup/users/kzqian/data/hubert/librispeech/km/km_100_7 /nobackup/users/kzqian/data/hubert/librispeech/hubert_base_ls960.pt

set -e
nshard=$1
layer=$2
meta_dir=$3
feat_dir=$4
lab_dir=$5
km_dir=$6
ckpt_path=$7


ngpus=$(nvidia-smi -L | wc -l)
rrr=$(bc <<< "${nshard}%${ngpus}")
if [ "$rrr" -ne "0" ]; then
   echo "Invalid number of shards!";
   exit;
fi
ntasks=$(bc <<< "${nshard}/${ngpus}")
if [ "$ntasks" -gt "8" ]; then
   echo "Number of tasks too large!";
   exit;
fi


mkdir -p -m 777 $feat_dir
pids=()
for rank in $(seq 0 $((nshard - 1))); do
CUDA_VISIBLE_DEVICES=$(bc <<< "${rank}/${ntasks}") python dump_hubert_feature_perm.py $meta_dir "train" $ckpt_path $layer ${nshard} ${rank} $feat_dir &
pids+=($!)
done
python dump_hubert_feature_perm.py $meta_dir "valid" $ckpt_path $layer 1 0 $feat_dir &
pids+=($!)
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} dump_feature jobs failed." && false


#mkdir -p -m 777 $(dirname $km_dir)
#python learn_kmeans.py $feat_dir "train" ${nshard} "${km_dir}" 100 --percent -1


mkdir -p -m 777 $lab_dir
pids=()
for rank in $(seq 0 $((nshard - 1))); do
CUDA_VISIBLE_DEVICES=$(bc <<< "${rank}/${ntasks}") python dump_km_label.py $feat_dir "train" "${km_dir}" ${nshard} ${rank} $lab_dir &
pids+=($!)
done
python dump_km_label.py $feat_dir "valid" "${km_dir}" 1 0 $lab_dir &
pids+=($!)
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} dump_label jobs failed." && false


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/valid_0_1.km $lab_dir/valid.km

for rank in $(seq 0 $((nshard - 1))); do
  rm $lab_dir/train_${rank}_${nshard}.km
done

#cp /nobackup/users/kzqian/data/hubert/librispeech/label_100_7/dict.km.txt $lab_dir/

#rm -rf $feat_dir


exit 1