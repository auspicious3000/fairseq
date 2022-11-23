import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.emergent.emergent import (
    HubertEncoder, 
    HubertAsrConfig,
)
from fairseq.tasks.emergent_pretraining import (
    EmergentPretrainingConfig,
    EmergentPretrainingTask,
)
from fairseq.pdb import set_trace

import logging
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig(FairseqDataclass):
    no_share_bhd: bool = field(
        default=False,
        metadata={"help": "if true, not share feature encoder"},
    )
    D_hid: int = field(
        default=768,
        metadata={"help": "dim hidden"},
    )
    D_emb: int = field(
        default=256,
        metadata={"help": "dim embedding"},
    )
    vocab_size: int = field(
        default=4035,
        metadata={"help": "vocab size"},
    )
    num_layers: int = field(
        default=1,
        metadata={"help": "num of rnn layers"},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability"},
    )
    temp: float = field(
        default=1.0,
        metadata={"help": "Gumbel temperature"},
    )
    hard: bool = field(
        default=False,
        metadata={"help": "Hard Gumbel-Softmax Sampling"},
    )
    seq_len: int = field(
        default=30,
        metadata={"help": "max len"},
    )
    num_directions: int = field(
        default=1,
        metadata={"help": "num of rnn directions"},
    )
    unit_norm: bool = field(
        default=False,
        metadata={"help": "num of rnn layers"},
    )
    # this holds the loaded hubert args
    w2v_args: FairseqDataclass = HubertAsrConfig()


@register_model("single_agent", dataclass=AgentConfig)
class SingleAgent(BaseFairseqModel):
    def __init__(self, cfg: AgentConfig, task: EmergentPretrainingTask):
        super().__init__()
        
        self.no_share_bhd = cfg.no_share_bhd
        self.D_hid = cfg.D_hid
        self.num_dist = task.cfg.num_negs * 2
        
        if self.no_share_bhd:
            logger.info("Not sharing feature encoder for each agent.")
            self.beholder1 = HubertEncoder(cfg.w2v_args, task)
            self.beholder2 = HubertEncoder(cfg.w2v_args, task)
        else:
            logger.info("Sharing feature encoder for each agent.")
            self.beholder = HubertEncoder(cfg.w2v_args, task)
            
        self.speaker = Speaker(cfg)
        self.listener = RnnListener(cfg)

    @classmethod
    def build_model(cls, cfg: AgentConfig, task: EmergentPretrainingTask):
        """Build a new model instance."""

        model = SingleAgent(cfg, task)
        return model

    def generate_ec(self, samples):
        with torch.no_grad():
            if self.no_share_bhd:
                hid = self.beholder1(samples)  # shared
            else:
                hid = self.beholder(samples)  # shared

            _, comm_action, _ = self.speaker(hid)
            
        return comm_action

    def forward(self, samples):
        samples_pos = samples['pos_audios']
        mask_pos = samples['padding_mask_pos']
        samples_neg = samples['neg_audios']
        mask_neg = samples['padding_mask_neg']
        
        if self.no_share_bhd:
            hid_pos = self.beholder1(samples_pos, mask_pos)  # shared
        else:
            hid_pos = self.beholder(samples_pos, mask_pos)  # shared
        hid_pos = hid_pos["encoder_out"]
        
        spk_logits, comm_action, spk_cap_len = self.speaker(hid_pos)
        
        if self.no_share_bhd:
            hid_neg = self.beholder2(samples_neg, mask_neg)
        else:
            hid_neg = self.beholder(samples_neg, mask_neg)
        hid_neg = hid_neg["encoder_out"]

        rnn_hid = self.listener(comm_action[:, :-1], spk_cap_len-1, spk_logits[:, :-1, :])
        
        result = {
            "listener_out": rnn_hid,
            "candidates": hid_neg,
            "spk_cap_len": spk_cap_len,
        }
        return result
    
    def get_logits(self, net_output):
        listener_out = net_output['listener_out'].unsqueeze(1).expand(-1, self.num_dist, -1)
        candidates = net_output['candidates'].view(-1, self.num_dist, self.D_hid)
        l2_diff_dist = torch.mean((listener_out - candidates)**2, 2).view(-1, self.num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        return l2_logits
    
    def get_targets(self, net_output):
        tmp = net_output['listener_out']
        return tmp.new_zeros(tmp.size(0), dtype=torch.long)
    
    def get_correct(self, logits, y):  # logits: [batch_size, num_of_classes]
        _, y_max_idx = torch.max(logits, 1)  # [batch_size]
        eq = torch.eq(y_max_idx, y)
        n_correct = float(eq.sum().item())
        return n_correct
    
    def get_length(self, net_output):
        spk_cap_len = net_output['spk_cap_len']
        msg_len = spk_cap_len.sum().item()
        return msg_len


class Speaker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)

        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)

        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.num_layers = args.num_layers
        self.drop = nn.Dropout(p=args.dropout)

        self.vocab_size = args.vocab_size
        self.temp = args.temp
        self.hard = args.hard
        self.seq_len = args.seq_len

    def forward(self, h_img):
        # h_img [batch_size, dim]
        batch_size = h_img.size(0)
        
        h_img = h_img.view(1, batch_size, self.D_hid).repeat(self.num_layers, 1, 1)
        
        initial_input = self.emb(2 * torch.ones([batch_size, 1], dtype=torch.int64, device=h_img.device))
        out_, hid_ = self.rnn(initial_input, h_img)
        logits_ = []
        labels_ = []
        for idx in range(self.seq_len):
            logit_ = self.hid_to_voc(out_.view(-1, self.D_hid))
            c_logit_, comm_label_ = gumbel_softmax(logit_, self.temp, self.hard, idx)
            input_ = torch.matmul(c_logit_.unsqueeze(1), self.emb.weight)
            out_, hid_ = self.rnn(input_, hid_)
            logits_.append(c_logit_.unsqueeze(1))
            labels_.append(comm_label_)
        
        logits_ = torch.cat(logits_, dim=1)
        labels_ = torch.cat(labels_, dim=-1)
        tmp = torch.zeros(logits_.size(-1))
        tmp[3] = 1
        logits_[:, -1, :] = tmp
        labels_[:, -1] = 3
        pad_g = ((labels_ == 3).cumsum(1) == 0)
        labels_ = pad_g * labels_
        pad_ = torch.zeros_like(logits_)
        pad_[:, :, 0] = 1
        
        logits_ = torch.where(pad_g.unsqueeze(-1).repeat(1, 1, logits_.size(-1)), logits_, pad_)

        cap_len = pad_g.cumsum(1).max(1).data + 1

        return logits_, labels_, cap_len
    
    
class RnnListener(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True, 
                          bidirectional=(args.num_directions==2))
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)
        self.hid_to_hid = nn.Linear(args.num_directions * args.D_hid, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers
        self.num_directions = args.num_directions
        self.vocab_size = args.vocab_size
        self.unit_norm = args.unit_norm

    def forward(self, spk_msg, spk_msg_lens, spk_logit):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size(0)

        spk_msg_emb = torch.matmul(spk_logit, self.emb.weight)
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = nn.utils.rnn.pack_padded_sequence(spk_msg_emb, spk_msg_lens.cpu(), batch_first=True,
                                                 enforce_sorted=False)
        _, h_n = self.rnn(pack)

        h_n = h_n[-self.num_directions:, :, :]
        out = h_n.transpose(0, 1).contiguous().view(batch_size, self.num_directions * self.D_hid)
        out = self.hid_to_hid(out)

        if self.unit_norm:
            norm = torch.norm(out, p=2, dim=1, keepdim=True).detach() + 1e-9
            out = out / norm.expand_as(out)

        return out


def gumbel_softmax_sample(logits, temp, idx_=10):
    y = (logits -torch.empty_like(logits).exponential_().log()) / temp
    if idx_ == 0:
        y[:, 3] = -float('Inf')
    return F.softmax(y, dim=-1)


def gumbel_softmax(logits, temp, hard, idx_=10):
    y_soft = gumbel_softmax_sample(logits, temp, idx_)
    _, y_max_idx = torch.max(y_soft, dim=1, keepdim=True)
    if hard:
        y_hard = torch.zeros_like(logits).scatter_(1, y_max_idx.data, 1)
        y = y_hard - y_soft.detach() + y_soft
    return y_soft, y_max_idx
