#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Optional

from torch import nn, Tensor
from . import Module


class Linear(nn.Linear):
    def __init__(self, input_size, label_num, dropout=None, bias=True):
        super().__init__(input_size, label_num, bias)

    def forward(self, embedding: Tensor, seq_lens: Tensor = None, gold: Optional = None):
        return super(Linear, self).forward(embedding)


Module.weak_register('Linear', Linear)
