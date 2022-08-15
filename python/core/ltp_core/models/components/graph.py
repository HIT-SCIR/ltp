#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from collections import namedtuple

from torch import nn

from ltp_core.models.nn.biaffine import Biaffine
from ltp_core.models.nn.mlp import MLP

GraphResult = namedtuple("GraphResult", ["arc_logits", "rel_logits", "attention_mask"])


class BiaffineClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_labels,
        dropout=0.1,
        arc_hidden_size=500,
        rel_hidden_size=100,
    ):
        super().__init__()

        self.label_num = num_labels
        self.input_size = input_size
        self.arc_hidden_size = arc_hidden_size
        self.rel_hidden_size = rel_hidden_size

        self.mlp_arc = MLP(
            [input_size, arc_hidden_size * 2],
            output_dropout=dropout,
            output_activation=nn.ReLU,
        )
        self.mlp_rel = MLP(
            [input_size, rel_hidden_size * 2],
            output_dropout=dropout,
            output_activation=nn.ReLU,
        )

        self.arc_atten = Biaffine(arc_hidden_size, arc_hidden_size, 1, bias_x=True, bias_y=False)
        self.rel_atten = Biaffine(
            rel_hidden_size, rel_hidden_size, num_labels, bias_x=True, bias_y=True
        )

    def forward(self, hidden_states, attention_mask=None):
        bs, seqlen = hidden_states.shape[:2]

        arc = self.mlp_arc(hidden_states)
        arc = arc.view(bs, seqlen, 2, self.arc_hidden_size)
        arc_h, arc_d = arc.unbind(axis=-2)

        rel = self.mlp_rel(hidden_states)
        rel = rel.view(bs, seqlen, 2, self.rel_hidden_size)
        rel_h, rel_d = rel.unbind(axis=-2)

        s_arc = self.arc_atten(arc_d, arc_h).squeeze_(1)
        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        return GraphResult(arc_logits=s_arc, rel_logits=s_rel, attention_mask=attention_mask)
