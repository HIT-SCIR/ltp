#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Callable, Union
from . import Optimizer


def PretrainedOptim(named_params, optim: Union[str, Callable], lr=1e-5, weight_decay=0.01,
                    bert_lr=None, bert_weight_decay=None, **kwargs):
    if bert_lr is None:
        bert_lr = lr
    if bert_weight_decay is None:
        bert_weight_decay = weight_decay
    param_optimizer = list(named_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = {'in_pretrained': {'decay': [], 'no_decay': []}, 'out_pretrained': {'decay': [], 'no_decay': []}}
    for n, p in param_optimizer:
        is_in_pretrained = 'in_pretrained' if 'pretrained' in n else 'out_pretrained'
        is_no_decay = 'no_decay' if any(nd in n for nd in no_decay) else 'decay'
        params[is_in_pretrained][is_no_decay].append(p)

    optimizer_grouped_parameters = [
        {'params': params['in_pretrained']['decay'], 'weight_decay': bert_weight_decay, 'lr': bert_lr},
        {'params': params['in_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': bert_lr},
        {'params': params['out_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
        {'params': params['out_pretrained']['no_decay'], 'weight_decay': weight_decay, 'lr': lr},
    ]

    assert optim != "PretrainedOptim" and optim != "BertAdamW"

    if isinstance(optim, str):
        return Optimizer.by_name(optim)(
            optimizer_grouped_parameters, lr=lr,
            weight_decay=weight_decay, **kwargs)
    elif isinstance(optim, Callable):
        return optim(
            optimizer_grouped_parameters, lr=lr,
            weight_decay=weight_decay, **kwargs)
    else:
        raise NotImplementedError(f"Optim not support {type(optim)}")


def BertAdamW(named_params, lr=1e-5, weight_decay=0.01, eps=1e-8, **kwargs):
    from transformers import optimization
    return PretrainedOptim(named_params, lr=lr, weight_decay=weight_decay, optim=optimization.AdamW, eps=eps, **kwargs)


Optimizer.weak_register("PretrainedOptim", PretrainedOptim)
Optimizer.weak_register("BertAdamW", BertAdamW)
