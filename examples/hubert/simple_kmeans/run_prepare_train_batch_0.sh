#!/bin/bash

set -e
head=$1
tail=$2

#rm -rf /nobackup/users/yangzhan/ssl-disentangle/feats_v05
for rank in $(seq $((head)) $((tail))); do
echo Preparing label $rank
./prepare_train_perm_batch.sh 4 12 "/nobackup/users/yangzhan/ssl-disentangle/feats_v05_1/feat_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/labels_v05/label_${rank}" "/nobackup/users/yangzhan/ssl-disentangle/km_v05/km_100_12" True
done

exit 1