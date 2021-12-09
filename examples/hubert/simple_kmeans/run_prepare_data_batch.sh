#!/bin/bash

set -e
head=$1
tail=$2


for rank in $(seq $((head)) $((tail))); do
echo Preparing label $rank
./prepare_valid_perm_batch.sh 4 12 "/nobackup/users/yangzhan/ssl-disentangle/feat_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/label_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/km_v03/km_100_12" "/nobackup/users/kzqian/data/hubert/librispeech/hubert_base_ls960.pt"
done

exit 1