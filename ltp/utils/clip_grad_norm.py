#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch.nn.utils import clip_grad_norm_


def clip_grad_norm(parameters, max_norm, norm_type=None, pretrained_norm=None):
    if max_norm is None:
        return
    if norm_type is None:
        norm_type = 2
    if pretrained_norm is None:
        clip_grad_norm_(filter(lambda p: p.requires_grad, parameters), max_norm, norm_type)
    else:
        in_pretrained = []
        out_pretrained = []
        for n, p in parameters:
            if not p.requires_grad:
                continue
            if 'pretrained' in n:
                in_pretrained.append(p)
            else:
                out_pretrained.append(p)
        clip_grad_norm_(in_pretrained, max_norm, norm_type)
        clip_grad_norm_(out_pretrained, pretrained_norm, norm_type)
