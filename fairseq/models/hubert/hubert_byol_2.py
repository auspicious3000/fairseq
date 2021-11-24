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
import torch.nn.functional as F
from dataclasses import dataclass, field
import fairseq
# from fairseq import utils
# from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
# from fairseq.models.wav2vec.wav2vec2 import (
#     ConvFeatureExtractionModel
# )
# from fairseq.models.wav2vec.wav2vec2_1 import TransformerEncoder_1
from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer
from fairseq.models.hubert.hubert_1 import HubertModel_1, HubertConfig_1
# from fairseq.models.hubert.hubert import HubertModel, HubertConfig
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from omegaconf import II, MISSING
# from fairseq.pdb import set_trace

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

@dataclass
class HubertConfig_2(HubertConfig_1):

    max_update: int = field(
        default=400000, metadata={"help": "max number of updates"}
    )

    base_target_ema: float = field(
        default=0.999, metadata={"help": "max number of updates"}
    )

    update_target_frequency: int = field(
        default=1, metadata={"help": "update target for this many updates"}
    )

    pretrained_hubert_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )

    load_from_pretrained: bool = field(
        default=False, metadata={"help": "load hubert from pretrained"}
    )

    projector_hidden_size: int = field(
        default=4096, metadata={"help": "projector hidden size"}
    )

    predictor_hidden_size: int = field(
        default=4096, metadata={"help": "predictor hidden size"}
    )

    output_size: int = field(
        default=768, metadata={"help": "predictor outpu size of predictor and projector"}
    )


@register_model("hubertbyol2", dataclass=HubertConfig_2)
class HubertByol2(BaseFairseqModel):
    def __init__(
        self, 
        cfg: HubertConfig_2, 
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ):
        super().__init__()
        self.cfg = cfg
        self.hubertmodel_online = HubertModel_1(cfg, task_cfg, dictionaries)
        self.hubertmodel_target = HubertModel_1(cfg, task_cfg, dictionaries)

        if cfg.load_from_pretrained:
            pretrained_hubert_path = cfg.pretrained_hubert_path
            models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_hubert_path])
            model = models[0]
            hubert_state_dict = {k: v for k, v in model.state_dict().items() if 'label_embs_concat' != k}
            missing_keys_online, unexpected_keys_online = self.hubertmodel_online.load_state_dict(hubert_state_dict, strict=False)
            print('online model: missing_keys', missing_keys_online, 'unexpected_keys', unexpected_keys_online)
            missing_keys_target, unexpected_keys_target = self.hubertmodel_target.load_state_dict(hubert_state_dict, strict=False)
            print('target model: missing_keys', missing_keys_target, 'unexpected_keys', unexpected_keys_target)

        print('fix target model...')
        for parameter in self.hubertmodel_target.parameters():
            parameter.requires_grad = False

        self.projector_online = MLP(cfg.encoder_embed_dim, cfg.projector_hidden_size, cfg.output_size)
        self.predictor_online = MLP(cfg.encoder_embed_dim, cfg.projector_hidden_size, cfg.output_size)
        self.projector_target = MLP(cfg.encoder_embed_dim, cfg.projector_hidden_size, cfg.output_size)

        self.max_update = cfg.max_update
        self.base_target_ema = cfg.base_target_ema
        self.update_target_frequency = cfg.update_target_frequency

        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: HubertConfig_2, task: HubertPretrainingTask):
        """Build a new model instance."""
        return HubertByol2(cfg, task.cfg, task.dictionaries)

    def compute_decay(self, num_updates):
        # from BYOL github code https://github.com/deepmind/deepmind-research/blob/2c7c401024c42c4fb1aa20a8b0471d2e6b480906/byol/utils/schedules.py#L20
        def _cosine_decay(global_step, max_steps, initial_value):
            """Simple implementation of cosine decay from TF1."""
            global_step = min(global_step, max_steps)
            cosine_decay_value = 0.5 * (1 + np.cos(np.pi * global_step / max_steps))
            decayed_learning_rate = initial_value * cosine_decay_value
            return decayed_learning_rate
        decay = _cosine_decay(num_updates, self.max_update, 1.)
        return 1. - (1. - self.base_target_ema) * decay

    def merge_state_dict(self, sd1, sd2, tau):
        merged_state_dict = {}
        for k in sd1:
            merged_state_dict[k] = (tau * sd1[k] + (1-tau) * sd2[k]).detach()
        return merged_state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

        tau = self.compute_decay(num_updates)

        if self.num_updates % self.update_target_frequency == 0:
            # merge online and target hubert model
            state_dict_online = self.hubertmodel_online.state_dict()
            state_dict_target = self.hubertmodel_target.state_dict()
            merged_state_dict = self.merge_state_dict(state_dict_target, state_dict_online, tau)
            missing_keys_target, unexpected_keys_target = self.hubertmodel_target.load_state_dict(merged_state_dict, strict=True)
            assert not missing_keys_target, f'missing_keys_target {missing_keys_target} is not empty'
            assert not unexpected_keys_target, f'unexpected_keys_target {unexpected_keys_target} is not empty'
            # merge online and target projector
            state_dict_online = self.projector_online.state_dict()
            state_dict_target = self.projector_target.state_dict()
            merged_state_dict_projector = self.merge_state_dict(state_dict_target, state_dict_online, tau)
            self.projector_target.load_state_dict(merged_state_dict_projector, strict=True)


    def compute_representation_loss(
        self, 
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        spk_emb: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ):
        # pass view 1 and view 2 to online and target networks respectively
        result_online = self.hubertmodel_online(source_1, spk_emb, target_list, padding_mask, mask, features_only, output_layer, detach_features=False)
        with torch.no_grad():
            result_target = self.hubertmodel_target(source_2, spk_emb, target_list, padding_mask, mask, features_only=True, output_layer=output_layer, detach_features=False)
        # project and predict using online embeddingg
        embedding_online, padding_mask_online = result_online['x_no_speaker_emb'], result_online['padding_mask']
        proj_out_online = self.projector_online(embedding_online)
        pred_out_online = self.predictor_online(proj_out_online)
        # project using target embeddingg
        embedding_target, padding_mask_target = result_target['x_no_speaker_emb'].detach(), result_target['padding_mask']
        assert torch.all(padding_mask_online == padding_mask_target)
        proj_out_target = self.projector_target(embedding_target)
        assert proj_out_online.shape == proj_out_target.shape, f'{proj_out_online.shape} == {proj_out_target.shape}'
        assert pred_out_online.shape == proj_out_target.shape, f'{pred_out_online.shape} == {proj_out_target.shape}'
        # normalize online predcitor output and target projector output
        online_repr = F.normalize(pred_out_online[~padding_mask_online.T], p=2.0, dim=-1, eps=1e-12) # -1 is the hidden dimension
        target_repr = F.normalize(proj_out_target[~padding_mask_target.T], p=2.0, dim=-1, eps=1e-12)
        # compute loss
        repr_loss = F.mse_loss(online_repr, target_repr, reduction='mean')
        
        return {'repr_loss': repr_loss, 'result_online': result_online}

    def forward(
        self,
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        spk_emb: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        result_1 = self.compute_representation_loss(source_1, source_2, spk_emb, target_list, padding_mask, mask, features_only, output_layer)
        repr_loss_1, result_online = result_1['repr_loss'], result_1['result_online']

        result_2 = self.compute_representation_loss(source_2, source_1, spk_emb, target_list, padding_mask, mask, features_only=True, output_layer=output_layer)
        repr_loss_2 = result_2['repr_loss']

        repr_loss = repr_loss_1 + repr_loss_2
        result_online['repr_loss'] = repr_loss
        return result_online


    def get_logits(self, net_output, is_masked=True):
        return self.hubertmodel_online.get_logits(net_output, is_masked)
    
    def get_targets(self, net_output, is_masked=True):
        return self.hubertmodel_online.get_targets(net_output, is_masked)

    def get_extra_losses(self, net_output):

        extra_losses, names = self.hubertmodel_online.get_extra_losses(net_output)
        if "repr_loss" in net_output:
            extra_losses.append(net_output["repr_loss"])
            names.append("repr_loss")

        return extra_losses, names

class MLP(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_channels, mlp_hidden_size, )
        self.bn = nn.BatchNorm1d(mlp_hidden_size, )
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(mlp_hidden_size, projection_size)

    def forward(self, x):
        x = self.l1(x).transpose(1,2)
        x = self.bn(x).transpose(1,2)
        x = self.relu(x)
        x = self.l2(x)
        return x