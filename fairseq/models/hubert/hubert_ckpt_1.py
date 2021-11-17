# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import contextlib
# from argparse import Namespace
# from typing import Any

# import torch
# import torch.nn as nn
# from dataclasses import dataclass, field
# from fairseq import checkpoint_utils, tasks, utils
# from fairseq.dataclass import FairseqDataclass
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
# from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
# from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
# from fairseq.models.hubert.hubert_1 import HubertModel_1
# from fairseq.tasks import FairseqTask
# from omegaconf import II, MISSING

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
import fairseq
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel
)
from fairseq.models.wav2vec.wav2vec2_1 import TransformerEncoder_1
from fairseq.models.hubert.hubert_1 import HubertModel_1, HubertConfig_1
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from omegaconf import II, MISSING
from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

@dataclass
class HubertCkpt1Config(HubertConfig_1):
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune hubert for this many updates"}
    )

    pretrained_hubert_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )


@register_model("hubertckpt1", dataclass=HubertCkpt1Config)
class HubertCkpt1(BaseFairseqModel):
    def __init__(
        self, 
        cfg: HubertCkpt1Config, 
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ):
        super().__init__()
        self.cfg = cfg
        self.hubertmodel_1 = HubertModel_1(cfg, task_cfg, dictionaries)

        pretrained_hubert_path = cfg.pretrained_hubert_path
        models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_hubert_path])
        model = models[0]
        hubert_state_dict = {k: v for k, v in model.state_dict().items() if 'label_embs_concat' != k}
        missing_keys, unexpected_keys = self.hubertmodel_1.load_state_dict(hubert_state_dict, strict=False)
        print('missing_keys', missing_keys, 'unexpected_keys', unexpected_keys)
        self.missing_keys, self.unexpected_keys = missing_keys, unexpected_keys

        self.num_updates = 0

        self.pretrained_freezed = False # a flag denoting if the pretrained hubert paramters are freezed
        self.freeze_pretrained_paramters()
        assert self.pretrained_freezed == True
        
        self.freeze_finetune_updates = self.cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg: HubertCkpt1Config, task: HubertPretrainingTask):
        """Build a new model instance."""
        return HubertCkpt1(cfg, task.cfg, task.dictionaries)

    def release_all_parameters(self):
        if self.pretrained_freezed:
            print('release all parameters')
            for n, parameter in self.hubertmodel_1.named_parameters():
                parameter.requires_grad = True
        self.pretrained_freezed = False
        
    def freeze_pretrained_paramters(self):
        for n, parameter in self.hubertmodel_1.named_parameters():
            if n not in self.missing_keys: 
                # if paramter names are in pretrained model, do not update their parameter 
                parameter.requires_grad = False
            else:
                print('do not freeze parameter:', n)
                parameter.requires_grad = True
        self.pretrained_freezed = True

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    

    def upgrade_state_dict_named(self, state_dict, name):
        return self.hubertmodel_1.upgrade_state_dict_named(state_dict, name)
    
    def apply_mask(self, x, padding_mask, target_list):
        return self.hubertmodel_1.apply_mask(x, padding_mask, target_list)
    
    def compute_nce(self, x, pos, negs):
        return self.hubertmodel_1.compute_nce(x, pos, neg)
    
    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        return self.hubertmodel_1.forward_features(source)
    
    def forward_targets(
        self, features: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hubertmodel_1.forward_targets(features, target_list)
    
    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.hubertmodel_1.forward_padding_mask(features, padding_mask)

    def forward(
        self,
        source: torch.Tensor,
        spk_emb: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        finetune_hubert = self.num_updates >= self.freeze_finetune_updates
        if finetune_hubert:
            self.release_all_parameters()
            assert not self.pretrained_freezed

        return self.hubertmodel_1(source, spk_emb, target_list, padding_mask, mask, features_only, output_layer)
    
    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hubertmodel_1.extract_features(source, padding_mask, mask, ret_conv, output_layer)

    def get_logits(self, net_output, is_masked=True):
        return self.hubertmodel_1.get_logits(net_output, is_masked)
    
    def get_targets(self, net_output, is_masked=True):
        return self.hubertmodel_1.get_targets(net_output, is_masked)

    def get_extra_losses(self, net_output):
        return self.hubertmodel_1.get_extra_losses(net_output)

    def remove_pretraining_modules(self):
        return self.hubertmodel_1.remove_pretraining_modules(net_output)
