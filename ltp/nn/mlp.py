#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch import nn


def MLP(layer_sizes, dropout=0.0, activation=nn.ReLU, output_dropout=None, output_activation=None):
    layers = []
    num_layers = len(layer_sizes) - 1
    for index in range(num_layers):
        if index < num_layers:
            layers.extend([
                nn.Linear(layer_sizes[index], layer_sizes[index + 1]),
                activation(),
                nn.Dropout(p=dropout)
            ])
        else:
            layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))

            if output_activation is not None:
                layers.append(output_activation())
            if output_activation is not None:
                layers.append(nn.Dropout(p=output_dropout))
    return nn.Sequential(*layers)
