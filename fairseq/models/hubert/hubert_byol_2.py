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
class Hubert2Config(HubertConfig_1):

    # label_embs_concat_warmup_updates: int = field(
    #     default=0, metadata={"help": "finetune label_embs_concat for this many updates"}
    # )

    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune hubert for this many updates"}
    )

    pretrained_hubert_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )

    max_update: int = field(
        default=400000, metadata={"help": "max number of updates"}
    )

    base_target_ema: float = field(
        default=0.999, metadata={"help": "max number of updates"}
    )

    update_target_frequency: int = field(
        default=1, metadata={"help": "update target for this many updates"}
    )

@register_model("hubertbyol2", dataclass=Hubert2Config)
class Hubert2(BaseFairseqModel):
    def __init__(
        self, 
        cfg: Hubert2Config, 
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ):
        super().__init__()
        self.cfg = cfg
        self.hubertmodel_online = HubertModel_1(cfg, task_cfg, dictionaries)
        self.hubertmodel_target = HubertModel_1(cfg, task_cfg, dictionaries)

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

        self.missing_keys_online, self.unexpected_keys_online = missing_keys_online, unexpected_keys_online
        self.missing_keys_target, self.unexpected_keys_target = missing_keys_target, unexpected_keys_target

        self.projector = TransformerSentenceEncoderLayer(
            embedding_dim=cfg.encoder_embed_dim,
            ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
            num_attention_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            activation_fn=cfg.activation_fn,
            layer_norm_first=cfg.layer_norm_first,
        )

        self.predictor = TransformerSentenceEncoderLayer(
            embedding_dim=cfg.encoder_embed_dim,
            ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
            num_attention_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            activation_fn=cfg.activation_fn,
            layer_norm_first=cfg.layer_norm_first,
        )

        self.max_update = cfg.max_update
        self.base_target_ema = cfg.base_target_ema
        self.update_target_frequency = cfg.update_target_frequency
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        # self.label_embs_concat_warmup_updates = cfg.label_embs_concat_warmup_updates

        self.freeze_pretrained_paramters()
        assert self.pretrained_freezed == True

        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: Hubert2Config, task: HubertPretrainingTask):
        """Build a new model instance."""
        return Hubert2(cfg, task.cfg, task.dictionaries)

    def compute_decay(self, num_updates):
        def _cosine_decay(global_step, max_steps, initial_value):
            """Simple implementation of cosine decay from TF1."""
            global_step = min(global_step, max_steps)
            cosine_decay_value = 0.5 * (1 + np.cos(np.pi * global_step / max_steps))
            decayed_learning_rate = initial_value * cosine_decay_value
            return decayed_learning_rate
        decay = _cosine_decay(num_updates, self.max_update, 1.)
        return 1. - (1. - self.base_target_ema) * decay

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

        tau = self.compute_decay(num_updates)

        if self.num_updates % self.update_target_frequency == 0:
            state_dict_online = {k: v for k, v in self.hubertmodel_online.state_dict().items()}
            state_dict_target = {k: v for k, v in self.hubertmodel_target.state_dict().items()}
            merged_state_dict = {}
            for k in state_dict_online:
                merged_state_dict[k] = (tau * state_dict_target[k] + (1-tau) * state_dict_online[k]).detach()
            missing_keys_target, unexpected_keys_target = self.hubertmodel_target.load_state_dict(merged_state_dict, strict=True)
            assert not missing_keys_target, f'missing_keys_target {missing_keys_target} is not empty'
            assert not unexpected_keys_target, f'unexpected_keys_target {unexpected_keys_target} is not empty'

    def release_all_parameters(self):
        if self.pretrained_freezed:
            print('release all parameters')
            for n, parameter in self.hubertmodel_online.named_parameters():
                parameter.requires_grad = True
        self.pretrained_freezed = False
    
    def freeze_pretrained_paramters(self):
        for n, parameter in self.hubertmodel_online.named_parameters():
            if n not in self.missing_keys_online: 
                # if paramter names are in pretrained model, do not update their parameter 
                parameter.requires_grad = False
            else:
                print('do not freeze parameter:', n)
                parameter.requires_grad = True
        self.pretrained_freezed = True

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
        result_online = self.hubertmodel_online(source_1, spk_emb, target_list, padding_mask, mask, features_only, output_layer, detach_features=False)
        with torch.no_grad():
            result_target = self.hubertmodel_target(source_2, spk_emb, target_list, padding_mask, mask, features_only=True, output_layer=output_layer)
        embedding_online, padding_mask_online = result_online['x_no_speaker_emb'], result_online['padding_mask']

        proj_out, _ = self.projector(embedding_online, self_attn_padding_mask=padding_mask_online, need_weights=False)
        pred_out, _ = self.predictor(proj_out, self_attn_padding_mask=padding_mask_online, need_weights=False)

        embedding_target, padding_mask_target = result_target['x_no_speaker_emb'].detach(), result_target['padding_mask']
        assert torch.all(padding_mask_online == padding_mask_target)

        online_repr = F.normalize(pred_out[~padding_mask_online.T], p=2.0, dim=-1, eps=1e-12) # -1 is the hidden dimension
        temp = (online_repr**2).sum(dim=-1)

        target_repr = F.normalize(embedding_target[~padding_mask_target.T], p=2.0, dim=-1, eps=1e-12)
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
        finetune_hubert = self.num_updates >= self.freeze_finetune_updates
        if finetune_hubert:
            self.release_all_parameters()
            assert not self.pretrained_freezed
        if finetune_hubert:

            result_1 = self.compute_representation_loss(source_1, source_2, spk_emb, target_list, padding_mask, mask, features_only, output_layer)
            repr_loss_1, result_online = result_1['repr_loss'], result_1['result_online']

            result_2 = self.compute_representation_loss(source_2, source_1, spk_emb, target_list, padding_mask, mask, features_only=True, output_layer=output_layer)
            repr_loss_2 = result_2['repr_loss']

            repr_loss = repr_loss_1 + repr_loss_2
            result_online['repr_loss'] = repr_loss
            return result_online
        
        else:
            result_online = self.hubertmodel_online(source_1, spk_emb, target_list, padding_mask, mask, features_only, output_layer, detach_features=False)
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
