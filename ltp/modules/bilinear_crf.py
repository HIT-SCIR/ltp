#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch import Tensor
from ltp.nn import MLP
from ltp import nn

from . import Module


class BiLinearCRF(Module):
    def __init__(self, input_size, label_num, dropout: float = 0.2, hidden_size=None, **kwargs):
        super().__init__(input_size, label_num, dropout)
        activation = kwargs.pop('activation', {'LeakyReLU': {}})
        self.mlp_rel_h = MLP(input_size, hidden_size, dropout=dropout, **activation)
        self.mlp_rel_d = MLP(input_size, hidden_size, dropout=dropout, **activation)
        self.biaffine = nn.Bilinear(hidden_size, hidden_size, label_num, bias_x=True, bias_y=True, expand=True)

        self.crf = nn.CRF(label_num)

    def forward(self, inputs: Tensor, length: Tensor, gold=None):
        rel_h = self.mlp_rel_h(inputs)
        rel_d = self.mlp_rel_d(inputs)

        logits = self.biaffine(rel_h, rel_d).permute(0, 2, 3, 1)
        return logits, length, self.crf
