from torch.utils import data
import torch
import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F

import tqdm
from npy_append_array import NpyAppendArray
from feature_utils import get_shard_range

import numpy as np
import pdb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature_batch_loader")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, split, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], 
                                                                  {'extractor_mode': 'group_norm_masked',
                                                                   'encoder_layers_1': 0},
                                                                  strict=False)
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk        
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def get_feats(self, uttrs, lengths):
        uttrs = uttrs.cuda()
        lengths = lengths.cuda()
        padding_mask = (~sequence_mask(lengths)).long()
        with torch.no_grad():
            feat_chunk, _ = self.model.extract_features(
                source=uttrs,
                spk_emb=0,
                padding_mask=padding_mask,
                mask=False,
                output_layer=self.layer,
            )
        return feat_chunk


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, tsv_dir, split, nshard, rank, max_chunk):
        
        with open(f"{tsv_dir}/{split}.tsv", "r") as f:
            self.root = f.readline().rstrip()
            lines = [line.rstrip() for line in f]
            start, end = get_shard_range(len(lines), nshard, rank)
            self.lines = lines[start:end]
            
        self.num_tokens = len(self.lines)  
        self.max_chunk = max_chunk
        
        logger.info(f'Finished init the {self.num_tokens} Utterances nshard_{nshard} rank_{rank} dataset...')
        
    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == 16000, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav
        
    def __getitem__(self, index):
        
        line = self.lines[index]
        subpath, nsample = line.split("\t")
        
        wav = self.read_audio(f"{self.root}/{subpath}", int(nsample))
        wav = wav.astype(np.float32)
        
        assert abs(len(wav) - int(nsample)) < 3
        
        return wav, len(wav)
    
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    def collate_fn(self, batch):
        l_max = 0
        new_batch = []
        for token in batch:
            uttr, l_uttr = token
            if l_uttr > l_max:
                l_max = l_uttr
            uttr_pad = np.pad(uttr, (0,self.max_chunk-l_uttr), 'constant')
            new_batch.append( (uttr_pad, l_uttr) ) 
            
        batch = new_batch 
        a, b = zip(*batch)
        
        uttr_pad = torch.from_numpy(np.stack(a, axis=0))[:,:l_max]
        length = torch.from_numpy(np.stack(b, axis=0))
        
        return uttr_pad, length
        

def worker_init_fn(x):
    return np.random.seed((torch.initial_seed()) % (2**32))   


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def dump_feature(reader, generator, num, split, nshard, rank, feat_dir, disable_tqdm=False):
    data_iter = iter(generator)

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for uttrs, lengths in tqdm.tqdm(data_iter, total=num, disable=disable_tqdm):
            feat_chunk = reader.get_feats(uttrs, lengths)
            len_out = torch.div((lengths-400), 320, rounding_mode='floor') + 1
            for feat_, len_ in zip(feat_chunk, len_out):
                feat_ = feat_[:len_, :]
                feat_f.append(feat_.cpu().numpy())
                leng_f.write(f"{len_}\n")
    logger.info("finished successfully")


def main(tsv_dir, split, ckpt_path, layer, nshard, rank, batch_size, num_workers, feat_dir, max_chunk, disable_tqdm):
    reader = HubertFeatureReader(ckpt_path, split, layer, max_chunk)
    
    dataset = Utterances(tsv_dir, split, nshard, rank, max_chunk)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=dataset.collate_fn)
    num = len(dataset) // batch_size
    dump_feature(reader, data_loader, num, split, nshard, rank, feat_dir, disable_tqdm)
    
         
def str2bool(v):
    return v.lower() in ('true')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("num_workers", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--disable_tqdm", type=str2bool, default=False)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))