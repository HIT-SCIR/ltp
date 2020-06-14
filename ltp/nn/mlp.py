#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from functools import partial
from torch import nn
import ltp.nn


def MLP(input_size, hidden_size, dropout, **kwargs):
    activation_func = ltp.nn.ReLU
    for activation, args in kwargs.items():
        activation_func = partial(getattr(ltp.nn.activation, activation), **args)

    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        activation_func(),
        nn.Dropout(p=dropout)
    )
