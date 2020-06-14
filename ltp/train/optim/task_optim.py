#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from . import Optimizer


def BertAdamW4CRF(named_params, lr=1e-5, weight_decay=0.01,
                  bert_lr=1e-5, bert_weight_decay=None,
                  crf_lr=1e-3, crf_weight_decay=None,
                  **kwargs):
    if bert_lr is None:
        bert_lr = lr
    if bert_weight_decay is None:
        bert_weight_decay = weight_decay
    if crf_weight_decay is None:
        crf_weight_decay = weight_decay

    param_optimizer = list(named_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = {'in_pretrained': {'decay': [], 'no_decay': []}, 'out_pretrained': {'crf': [], 'no_crf': []}}
    for n, p in param_optimizer:
        is_in_pretrained = 'in_pretrained' if 'pretrained' in n else 'out_pretrained'
        is_no_decay = 'no_decay' if any(nd in n for nd in no_decay) else 'decay'
        is_crf = 'crf' if 'crf' in n else 'no_crf'

        if is_in_pretrained == 'in_pretrained':
            params[is_in_pretrained][is_no_decay].append(p)
        else:
            params[is_in_pretrained][is_crf].append(p)

    optimizer_grouped_parameters = [
        {'params': params['in_pretrained']['decay'], 'weight_decay': bert_weight_decay, 'lr': bert_lr},
        {'params': params['in_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': bert_lr},
        {'params': params['out_pretrained']['crf'], 'weight_decay': crf_weight_decay, 'lr': crf_lr},
        {'params': params['out_pretrained']['no_crf'], 'weight_decay': weight_decay, 'lr': lr},
    ]

    from transformers import optimization
    return optimization.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, **kwargs)


def Lamb4CRF(named_params, lr=1e-5, weight_decay=0.01,
             bert_lr=1e-5, bert_weight_decay=None,
             crf_lr=1e-3, crf_weight_decay=None,
             **kwargs):
    if bert_lr is None:
        bert_lr = lr
    if bert_weight_decay is None:
        bert_weight_decay = weight_decay
    if crf_weight_decay is None:
        crf_weight_decay = weight_decay

    param_optimizer = list(named_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = {'in_pretrained': {'decay': [], 'no_decay': []}, 'out_pretrained': {'crf': [], 'no_crf': []}}
    for n, p in param_optimizer:
        is_in_pretrained = 'in_pretrained' if 'pretrained' in n else 'out_pretrained'
        is_no_decay = 'no_decay' if any(nd in n for nd in no_decay) else 'decay'
        is_crf = 'crf' if 'crf' in n else 'no_crf'

        if is_in_pretrained == 'in_pretrained':
            params[is_in_pretrained][is_no_decay].append(p)
        else:
            params[is_in_pretrained][is_crf].append(p)

    optimizer_grouped_parameters = [
        {'params': params['in_pretrained']['decay'], 'weight_decay': bert_weight_decay, 'lr': bert_lr},
        {'params': params['in_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': bert_lr},
        {'params': params['out_pretrained']['crf'], 'weight_decay': crf_weight_decay, 'lr': crf_lr},
        {'params': params['out_pretrained']['no_crf'], 'weight_decay': weight_decay, 'lr': lr},
    ]

    from torch_optimizer import Lamb
    return Lamb(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, **kwargs)


Optimizer.weak_register("BertAdamW4CRF", BertAdamW4CRF)
Optimizer.weak_register("Lamb4CRF", Lamb4CRF)
