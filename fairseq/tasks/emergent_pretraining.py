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
from fairseq.data import Dictionary, EmergentDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class EmergentPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
        
    fine_tuning: bool = field(
        default=True, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
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
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    num_negs: Optional[int] = field(
        default=None,
        metadata={"help": "number of negative samples"},
    )
    len_crop: Optional[int] = field(
        default=None,
        metadata={"help": "crop length"},
    )


@register_task("emergent_pretraining", dataclass=EmergentPretrainingConfig)
class EmergentPretrainingTask(FairseqTask):

    cfg: EmergentPretrainingConfig

    def __init__(
        self,
        cfg: EmergentPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"EmergentPretrainingTask Config {cfg}")

        self.cfg = cfg

    @classmethod
    def setup_task(
        cls, cfg: EmergentPretrainingConfig, **kwargs
    ) -> "EmergentPretrainingTask":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        
        self.datasets[split] = EmergentDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            random_crop=self.cfg.random_crop,
            num_negs=self.cfg.num_negs,
            len_crop=self.cfg.len_crop,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
    
    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary
