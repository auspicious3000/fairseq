#!/bin/bash

set -e
head=$1
tail=$2

for rank in $(seq $((head)) $((tail))); do
echo Preparing label $rank
./prepare_valid_perm_batch.sh 4 12 "/nobackup/users/lrchai/ssl-disentangle/feats_v05/feat_${rank}" "/nobackup/users/lrchai/ssl-disentangle/labels_v05/label_valid/label_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/km_v05/km_100_12" True
done

exit 1