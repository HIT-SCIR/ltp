#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Optional
from torch import Tensor, nn

from ltp.modules import Module
from ltp import nn as ltp_nn


class RelativeTransformer(Module):
    def __init__(self, input_size: int, label_num: int, dropout: float, **kwargs):
        super().__init__(input_size, label_num, dropout)
        kwargs.setdefault('dropout', dropout)

        self.transformer = ltp_nn.RelativeTransformer(input_size, **kwargs)
        self.mlp = nn.Linear(input_size, label_num)

    def forward(self, inputs: Tensor, length: Tensor = None, gold: Optional = None):
        inputs, length, gold = self.transformer(inputs, length, gold)
        return self.mlp(inputs)
