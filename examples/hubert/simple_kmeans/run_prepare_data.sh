#!/bin/bash

set -e
head=$1
tail=$2


for rank in $(seq $((head)) $((tail))); do
echo Preparing label $rank
./prepare_data_perm.sh 40 12 "/nobackup/users/lrchai/ssl-disentangle/feat_${rank}" "/nobackup/users/lrchai/ssl-disentangle/label_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/km_v03/km_100_12" "/nobackup/users/kzqian/data/hubert/librispeech/hubert_base_ls960.pt"
done

exit 1