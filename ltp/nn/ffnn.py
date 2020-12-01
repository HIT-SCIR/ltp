#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from functools import partial
from torch import nn
from torch.nn.modules import activation


class FFNN(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_size=None, dropout=None, depth=1, last_dropout=False, **kwargs):
        activation_func = activation.LeakyReLU
        if hidden_size is None:
            hidden_size = output_size
        for activation_name, activation_args in kwargs.items():
            activation_func = partial(getattr(activation, activation_name), **activation_args)

        module = []
        for i in range(depth - 1):
            module.extend(
                [
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                    activation_func()
                ] + ([] if dropout is None else [nn.Dropout(p=dropout, inplace=True)])
            )
        module.append(nn.Linear(hidden_size if len(module) > 0 else input_size, output_size))
        if last_dropout:
            module.append(nn.Dropout(p=dropout, inplace=True))
        super().__init__(*module)
