#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from torch import nn, Tensor
from ltp.nn import Bilinear, ffnn


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias_x=True, bias_y=True, **kwargs):
        super(Biaffine, self).__init__()
        self.bw = nn.Parameter(torch.as_tensor([0.5, 0.5]), requires_grad=True)
        self.bilinear = Bilinear(in1_features, in2_features, out_features, bias_x=bias_x, bias_y=bias_y, expand=True)
        self.linear = ffnn(in1_features + in2_features, output_size=out_features, **kwargs.pop("linear", {}))

    def forward(self, x1: Tensor, x2: Tensor):
        len1, len2 = x1.size(1), x2.size(1)
        bw_norm = torch.softmax(self.bw, dim=-1)
        bilinear_logits = self.bilinear(x1, x2).permute(0, 2, 3, 1)
        x1 = x1.unsqueeze(2).contiguous().expand([-1, -1, len2, -1])
        x2 = x2.unsqueeze(1).contiguous().expand([-1, len1, -1, -1])
        linear_logits = self.linear(torch.cat([x1, x2], dim=-1))
        return bw_norm[0] * bilinear_logits + bw_norm[1] * linear_logits

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}, bias_x={}, bias_y={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias_x, self.bias_y
        )
