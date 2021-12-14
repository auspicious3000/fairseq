#!/bin/bash

#./prepare_valid_perm_batch.sh 4 12 "/nobackup/users/yangzhan/ssl-disentangle/feat_1" "/nobackup/users/yangzhan/ssl-disentangle/label_1" "/nobackup/users/yangzhan/ssl-disentangle/km_v03/km_100_12" False

set -e
nshard=$1
layer=$2
feat_dir=$3
lab_dir=$4
km_dir=$5
flag=$6

meta_dir="/nobackup/users/kzqian/data/hubert/librispeech/meta_grp"
spk2info="/nobackup/users/kzqian/data/hubert/librispeech/spk2info.dict"
ckpt_path="/nobackup/users/kzqian/ssl-disentangle/v05/checkpoints/checkpoint.best_loss_2.2779.pt"

ngpus=$(nvidia-smi -L | wc -l)
rrr=$(bc <<< "${nshard}%${ngpus}")
if [ "$rrr" -ne "0" ]; then
   echo "Invalid number of shards!";
   exit;
fi
ntasks=$(bc <<< "${nshard}/${ngpus}")
if [ "$ntasks" -gt "16" ]; then
   echo "Number of tasks too large!";
   exit;
fi


mkdir -p -m 777 $feat_dir
start=`date +%s`
pids=()
for rank in $(seq 0 $((nshard - 1))); do
CUDA_VISIBLE_DEVICES=$(bc <<< "${rank}/${ntasks}") python dump_hubert_feature_perm_batch.py $meta_dir $spk2info "valid" $ckpt_path $layer ${nshard} ${rank} 48 18 $feat_dir --disable_tqdm True &
pids+=($!)
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
end=`date +%s`
[ ${i} -gt 0 ] && echo "$0: ${i} dump_feature jobs failed." && false
echo Feature extraction time was `expr $end - $start` seconds.


#mkdir -p -m 777 $(dirname $km_dir)
#python learn_kmeans.py $feat_dir "train" ${nshard} "${km_dir}" 100 --percent -1


mkdir -p -m 777 $lab_dir
start=`date +%s`
pids=()
for rank in $(seq 0 $((nshard - 1))); do
CUDA_VISIBLE_DEVICES=$(bc <<< "${rank}/${ntasks}") python dump_km_label_batch.py $feat_dir "valid" "${km_dir}" ${nshard} ${rank} $lab_dir --disable_tqdm True &
pids+=($!)
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
end=`date +%s`
[ ${i} -gt 0 ] && echo "$0: ${i} dump_label jobs failed." && false
echo Label extraction time was `expr $end - $start` seconds.


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/valid_${rank}_${nshard}.km
done > $lab_dir/valid.km


for rank in $(seq 0 $((nshard - 1))); do
  rm $lab_dir/valid_${rank}_${nshard}.km
done


rm -rf $feat_dir

#exit 1