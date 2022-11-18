# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask
from fairseq.pdb import set_trace

from scipy import stats


@dataclass
class EmergentCriterionConfig(FairseqDataclass):
    pass


@register_criterion("emergent_criterion", dataclass=EmergentCriterionConfig)
class EmergentCriterion(FairseqCriterion):
    def __init__(
        self, cfg: EmergentCriterionConfig, task: FairseqTask):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(sample["net_input"])
        
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"
        
        logp = model.get_logits(net_output)
        targ = model.get_targets(net_output)
        loss_rtv = F.cross_entropy(logp, targ, reduction=reduction)
        loss += loss_rtv
        sample_size += targ.numel()
        
        ncorrect = model.get_correct(logp, targ)
        msg_len = model.get_length(net_output)
        
        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ncorrect": ncorrect,
            "sample_size": sample_size,
            "ntokens": sample_size,
            "nsentences": sample_size,
            "msg_len": msg_len,
        }
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("sample_size", sample_size)
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        metrics.log_scalar_sum("ncorrect", ncorrect)
        msg_len = sum(log.get("msg_len", 0) for log in logging_outputs) / sample_size
        metrics.log_scalar("msg_len", msg_len, round=2)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )       
        
        metrics.log_derived(
            "accuracy",
            lambda meters: safe_round(
                meters["ncorrect"].sum * 100.0 / meters["sample_size"].sum, 3
            )
            if meters["sample_size"].sum > 0
            else float("nan"),
        )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
