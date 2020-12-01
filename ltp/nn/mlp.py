#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from functools import partial
from torch import nn


def MLP(input_size, hidden_size, dropout, activation=nn.ReLU):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        activation(),
        nn.Dropout(p=dropout)
    )
