#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.modules.activation import *


class Swish(torch.nn.Module):
    r"""Swish activation function:

    .. math::
        \text{Swish}(x) = x * Sigmoid(x)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        return input * torch.sigmoid(input)


class HSwish(torch.nn.Module):
    r"""Hard Swish activation function:

    .. math::
        \text{Swish}(x) = x * \frac{ReLU6(x+3)}{6}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self):
        super().__init__()
        self.relu6 = ReLU6()

    def forward(self, input):
        return input * (self.relu6(input + 3) / 6)


class Mish(torch.nn.Module):
    r"""Mish activation function:

    .. math::
        \text{Mish}(x) = x * tanh(\ln(1 + e^x))

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        return input * (torch.tanh(F.softplus(input)))
