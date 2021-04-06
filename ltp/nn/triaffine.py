#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
# ref: https://github.com/yzhangcs/parser

import torch
import torch.nn as nn


class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y`.
    Usually, :math:`x` and :math:`y` can be concatenated with bias terms.
    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.
        - Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.
    Args:
        in_features (int):
            The size of the input feature.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
    .. _Efficient Second-Order TreeCRF for Neural Dependency Parsing:
        https://www.aclweb.org/anthology/2020.acl-main.302/
    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, in_features, bias_x=False, bias_y=False):
        super().__init__()

        self.in_features = in_features
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(in_features + bias_x, in_features, in_features + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.in_features}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, seq_len, seq_len, seq_len]``.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = torch.einsum('bzk,ikj->bzij', z, self.weight)
        # [batch_size, seq_len, seq_len, seq_len]
        s = torch.einsum('bxi,bzij,byj->bzxy', x, w, y)

        return s
