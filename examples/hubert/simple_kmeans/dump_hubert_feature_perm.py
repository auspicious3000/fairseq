# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F

from feature_utils import get_path_iterator, dump_feature

from fairseq.data.audio.audio_utils_1 import params2sos
from fairseq.data.audio.audio_utils_1 import change_gender
from scipy.signal import sosfilt
import pickle
import numpy as np
import pdb
Qmin, Qmax = 2, 5


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, spk2info, split, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        with open(spk2info, "rb") as f:
            spk2info = pickle.load(f)
        self.spk2info = spk2info[split]
        self.rng = np.random.default_rng()
        self.Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        
    def random_eq(self, wav, sr):
        z = self.rng.uniform(0, 1, size=(10,))
        Q = Qmin * (Qmax / Qmin)**z
        G = self.rng.uniform(-12, 12, size=(10,))
        sos = params2sos(G, self.Fc, Q, sr)
        wav = sosfilt(sos, wav)
        return wav
    
    def random_formant_f0(self, wav, sr, spk):
        _, (lo, hi, _) = self.spk2info[spk]
        
        ratio_fs = self.rng.uniform(1, 1.4)
        coin = (self.rng.random() > 0.5)
        ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
        
        ratio_ps = self.rng.uniform(1, 2)
        coin = (self.rng.random() > 0.5)
        ratio_ps = coin*ratio_ps + (1-coin)*(1/ratio_ps)
        
        ratio_pr = self.rng.uniform(1, 1.5)
        coin = (self.rng.random() > 0.5)
        ratio_pr = coin*ratio_pr + (1-coin)*(1/ratio_pr)
        
        if lo==50:
            lo=75
        if spk=="1447":
            lo, hi = 60, 400
        ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
        
        return ss

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        spk = path.split('/')[1]
        try:
            wav = self.random_formant_f0(wav, sr, spk)
        except UserWarning:
            print(f"Praat warining - {fileName}")
        except RuntimeError:
            print(f"Praat Error - {fileName}")
        wav = self.random_eq(wav, sr)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def main(tsv_dir, spk2info, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk):
    reader = HubertFeatureReader(ckpt_path, spk2info, split, layer, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("spk2info")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
