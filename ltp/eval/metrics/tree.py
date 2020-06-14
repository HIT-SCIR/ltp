#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Tuple

import torch
from . import Metric
from ltp.utils import length_to_mask


class TreeMetrics(Metric, alias='tree'):
    def __init__(self, eisner=False):
        """
        Tree Metric(LAS, UAS)

        :param pad_value: 被忽略的目标值
        """
        super(TreeMetrics, self).__init__(LAS=0., UAS=0.)
        self._eisner = eisner
        self._head_true = 0
        self._label_true = 0
        self._union_true = 0
        self._all = 0

    @property
    def UAS(self):
        return (self._head_true / self._all) if self._all != 0 else 0

    @property
    def LAS(self):
        return (self._union_true / self._all) if self._all != 0 else 0

    def step(self, y_pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]):
        arc_pred, label_pred, seq_len = y_pred
        mask = length_to_mask(seq_len + 1)
        mask[:, 0] = False
        if self._eisner:
            from ltp.utils import eisner
            arc_pred = eisner(arc_pred, mask)
        else:
            arc_pred = torch.argmax(arc_pred, dim=-1)
        label_pred = torch.argmax(label_pred, dim=-1)

        arc_real, label_real = y
        label_pred = label_pred.gather(-1, arc_pred.unsqueeze(-1)).squeeze(-1)

        mask = mask.narrow(-1, 1, mask.size(1) - 1)
        arc_pred = arc_pred.narrow(-1, 1, arc_pred.size(1) - 1)
        label_pred = label_pred.narrow(-1, 1, label_pred.size(1) - 1)

        head_true = (arc_pred == arc_real)[mask]
        label_true = (label_pred == label_real)[mask]

        self._head_true += torch.sum(head_true).item()
        self._label_true += torch.sum(label_true).item()
        self._union_true += torch.sum(label_true[head_true]).item()
        self._all += torch.sum(mask).item()

    def clear(self):
        self._head_true = 0
        self._label_true = 0
        self._union_true = 0
        self._all = 0

    def compute(self):
        return {'LAS': self.LAS, 'UAS': self.UAS}
