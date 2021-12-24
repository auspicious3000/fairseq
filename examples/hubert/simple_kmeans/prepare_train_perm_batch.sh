#!/bin/bash

#./prepare_valid_perm_batch.sh hd tl

set -e
head=$1
tail=$2
layer=12
meta_dir="/gpfs/u/scratch/LANG/LANGkzhq/data/hubert/librispeech/meta_des"
spk2info="/gpfs/u/scratch/LANG/LANGkzhq/data/hubert/librispeech/spk2info.dict"
ckpt_path="/gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/hubert_base_ls960.pt"
feat_dir="/gpfs/u/scratch/LANG/LANGkzhq/ssl-disentangle/feats_v05/train"
flag=True


for rank in $(seq $((head)) $((tail))); do
feat_subdir=$feat_dir/feat_${rank}
mkdir -p -m 777 $feat_subdir
python dump_hubert_feature_perm_batch.py $meta_dir $spk2info "train" $ckpt_path $layer 1 0 60 24 $feat_subdir --disable_tqdm $flag
done


#mkdir -p -m 777 $(dirname $km_dir)
#python learn_kmeans.py $feat_dir "train" ${nshard} "${km_dir}" 100 --percent -1


#mkdir -p -m 777 $lab_dir
#start=`date +%s`
#pids=()
#for rank in $(seq 0 $((nshard - 1))); do
#CUDA_VISIBLE_DEVICES=$(bc <<< "${rank}/${ntasks}") python dump_km_label_batch.py $feat_dir "valid" "${km_dir}" ${nshard} ${rank} $lab_dir --disable_tqdm True &
#pids+=($!)
#done
#i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#end=`date +%s`
#[ ${i} -gt 0 ] && echo "$0: ${i} dump_label jobs failed." && false
#echo Label extraction time was `expr $end - $start` seconds.


#for rank in $(seq 0 $((nshard - 1))); do
#  cat $lab_dir/valid_${rank}_${nshard}.km
#done > $lab_dir/valid.km


#for rank in $(seq 0 $((nshard - 1))); do
#  rm $lab_dir/valid_${rank}_${nshard}.km
#done


#rm -rf $feat_dir

#exit 1