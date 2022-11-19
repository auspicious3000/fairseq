# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io
from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)


def load_audio_spk(manifest_path, max_keep, min_keep, multiplier):
    n_long, n_short = 0, 0
    names, sizes = {}, []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                spk = items[0].split('/')[1]
                if spk in names.keys():
                    names[spk].append(items[0])
                else:
                    names[spk] = []
                sizes.append(sz)
    names = list(names.values())*multiplier
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}, "
            f"dataset_multiplier={multiplier}"
        )
    )
    return root, names, sizes


class EmergentDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        random_crop: bool = False,
        num_negs: int = 0,
        len_crop: int = 0,
        data_multiplier: int = 1,
    ):
        self.rng = np.random.default_rng()
        
        self.audio_root, self.audio_names, self.sizes = load_audio_spk(
            manifest_path, max_keep_sample_size, min_keep_sample_size,
            data_multiplier,
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.num_negs = num_negs
        self.len_crop = len_crop

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )
        #self.debug = []

    def get_audio(self, index):
        pos_idx = self.rng.integers(low=0, high=len(self.audio_names[index]), size=1)
        
        wav_path = os.path.join(self.audio_root, self.audio_names[index][pos_idx[0]])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        
        wavs = []
        head_candidates = np.arange(0, np.ceil((len(wav)-self.len_crop*self.sample_rate)/160), dtype=np.int64)
        heads = self.rng.choice(head_candidates, size=self.num_negs, 
                                replace=len(head_candidates)<self.num_negs)*160
        for head in heads:
            wavs.append(wav[head:head+self.len_crop*self.sample_rate])
        
        other_candidates = np.arange(0, len(self.audio_names[index]), dtype=np.int64)
        other_candidates = other_candidates[other_candidates!=pos_idx]
        others = self.rng.choice(other_candidates, size=self.num_negs, 
                                 replace=len(other_candidates)<self.num_negs)
        for other in others:
            wav_path = os.path.join(self.audio_root, self.audio_names[index][other])
            wav, cur_sample_rate = sf.read(wav_path)
            wav = torch.from_numpy(wav).float()
            wav = self.postprocess(wav, cur_sample_rate)
            head = self.rng.integers(0, len(wav)-self.len_crop*self.sample_rate)
            wavs.append(wav[head:head+self.len_crop*self.sample_rate])
            
        return wavs[0], wavs

    def __getitem__(self, index):
        #self.debug.append(index)
        #if len(self.debug)==2338:
        #    print(self.debug)
        pos_wav, wavs = self.get_audio(index)
        return {"id": index, "pos_wav": pos_wav, "wavs": wavs}

    def __len__(self):
        return len(self.audio_names)

    def collater(self, samples):
        pos_audios = [s["pos_wav"] for s in samples]
        neg_audios = [wv for s in samples for wv in s["wavs"]]
        if len(pos_audios) == 0:
            return {}
        
        collated_pos_audios, padding_mask_pos = self.collater_audio(
            pos_audios, self.len_crop*self.sample_rate
        )
        collated_neg_audios, padding_mask_neg = self.collater_audio(
            neg_audios, self.len_crop*self.sample_rate
        )
        
        net_input = {"pos_audios": collated_pos_audios, "padding_mask_pos": padding_mask_pos,
                     "neg_audios": collated_neg_audios, "padding_mask_neg": padding_mask_neg}
        batch = {
            "net_input": net_input,
        }
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        for i, audio in enumerate(audios):
            collated_audios[i] = audio
        
        return collated_audios, padding_mask
    
    def num_tokens(self, index):
        return self.len_crop*self.sample_rate

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
