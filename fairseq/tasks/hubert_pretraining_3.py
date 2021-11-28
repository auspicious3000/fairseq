# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset_1
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING
from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )


@dataclass
class HubertPretrainingConfig_3(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir_train: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_dir_valid: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    dict_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for dictionary in this directory instead",
        },
    )

    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    spk2info: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to spk2info",
        },
    )
    train_subset: str = field(
        default='',
        metadata={
            "help": "split name of training data, should be the same as that in data config",
        },
    )
    valid_subset: str = field(
        default='',
        metadata={
            "help": "split name of validation data, should be the same as that in data config",
        },
    )

    label_dir_pattern_train: Optional[str] = field(
        default='{}_{}',
        metadata={
            "help": "if set, looks for labels in this directory created using this string pattern",
        },
    )

    


@register_task("hubert_pretraining_3", dataclass=HubertPretrainingConfig_3)
class HubertPretrainingTask_3(FairseqTask):

    cfg: HubertPretrainingConfig_3

    def __init__(
        self,
        cfg: HubertPretrainingConfig_3,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask 3 Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

        assert bool(self.cfg.train_subset), f'train subset: {self.cfg.train_subset} cannot be empty'
        assert bool(self.cfg.valid_subset), f'valid subset: {self.cfg.valid_subset} cannot be empty'

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: HubertPretrainingConfig_3, **kwargs
    ) -> "HubertPretrainingTask_3":
        return cls(cfg)

    def load_dictionaries(self):
        assert self.cfg.dict_dir is not None
        dictionaries = [Dictionary.load(f"{self.cfg.dict_dir}/dict.{label}.txt") for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir_valid(self) -> str:
        if self.cfg.label_dir_valid is None:
            return self.cfg.data
        return self.cfg.label_dir_valid

    def get_label_dir_train(self, epoch) -> str:
        return self.cfg.label_dir_pattern_train.format(label_dir_train=self.cfg.label_dir_train, epoch=epoch)

    def has_sharded_data(self, split):
        return split == self.cfg.train_subset

    def load_dataset(self, split: str, **kwargs) -> None:
        if split == self.cfg.train_subset:
            assert 'epoch' in kwargs
            epoch = kwargs['epoch']
            paths = [
                f"{self.get_label_dir_train(epoch)}/{split}.{l}" for l in self.cfg.labels
            ]
            logger.info(f"loading training labels from {paths}")
        elif split == self.cfg.valid_subset:
            paths = [
                f"{self.get_label_dir_valid()}/{split}.{l}" for l in self.cfg.labels
            ]
            logger.info(f"loading validation labels from {paths}")
        else:
            raise ValueError(f'Invalid split: {split}')
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset_1(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            spk2info=self.cfg.spk2info
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
