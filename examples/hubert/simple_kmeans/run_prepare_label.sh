#!/bin/bash

head=$1
tail=$2

for rank in $(seq $((head)) $((tail))); do
./prepare_label.sh 4 7 /nobackup/users/junruin2/speech-resynthesis/LibriSpeech_8419_flat/tsv /nobackup/users/kzqian/data/hubert/librispeech/feat_8419_flat /nobackup/users/kzqian/data/hubert/librispeech/label_8419_flat_train_100_7_${rank} /nobackup/users/kzqian/data/hubert/librispeech/km_8419_flat_train/km100_7_${rank} /nobackup/users/kzqian/data/hubert/librispeech/hubert_base_ls960.pt
done

exit 1