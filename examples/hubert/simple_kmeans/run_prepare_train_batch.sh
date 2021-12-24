#!/bin/bash

set -e
head=$1
tail=$2

ngpus=$(nvidia-smi -L | wc -l)
if [ "$ngpus" -ne "6" ]; then
   echo "Invalid number of gpus!";
   exit;
fi

rrr=$(bc <<< "(${tail}-${head}+1)%${ngpus}")
if [ "$rrr" -ne "0" ]; then
   echo "Invalid partition!";
   exit;
fi

part=$(bc <<< "(${tail}-${head}+1)/${ngpus}")

start=`date +%s`
pids=()
for device in $(seq 0 5); do
hd=$(bc <<< "${head}+${device}*${part}")
tl=$(bc <<< "${head}+(${device}+1)*${part}-1")
echo Using GPU $device fer $hd to $tl
CUDA_VISIBLE_DEVICES=${device} ./prepare_train_perm_batch.sh $hd $tl &
pids+=($!)
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
end=`date +%s`
[ ${i} -gt 0 ] && echo "$0: ${i} dump_feature jobs failed." && false
echo Feature extraction time was `expr $end - $start` seconds.

exit 1