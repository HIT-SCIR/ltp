#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Optional
from torch import Tensor
from ltp.nn import MLP, Bilinear

from . import Module


class Graph(Module):
    def __init__(self, input_size, label_num, dropout, arc_hidden_size, rel_hidden_size, **kwargs):
        super(Graph, self).__init__(input_size, label_num, dropout)
        arc_dropout = kwargs.pop('arc_dropout', dropout)
        rel_dropout = kwargs.pop('rel_dropout', dropout)
        activation = kwargs.pop('activation', {})
        self.mlp_arc_h = MLP(input_size, arc_hidden_size, arc_dropout, **activation)
        self.mlp_arc_d = MLP(input_size, arc_hidden_size, arc_dropout, **activation)
        self.mlp_rel_h = MLP(input_size, rel_hidden_size, rel_dropout, **activation)
        self.mlp_rel_d = MLP(input_size, rel_hidden_size, rel_dropout, **activation)

        self.arc_atten = Bilinear(arc_hidden_size, arc_hidden_size, 1, bias_x=True, bias_y=False, expand=True)
        self.rel_atten = Bilinear(rel_hidden_size, rel_hidden_size, label_num, bias_x=True, bias_y=True, expand=True)

    def forward(self, embedding: Tensor, seq_lens: Tensor = None, gold: Optional = None):
        arc_h = self.mlp_arc_h(embedding)
        arc_d = self.mlp_arc_d(embedding)

        rel_h = self.mlp_rel_h(embedding)
        rel_d = self.mlp_rel_d(embedding)

        s_arc = self.arc_atten(arc_d, arc_h).squeeze_(1)
        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel, seq_lens
