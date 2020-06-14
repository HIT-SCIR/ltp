#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import torch
from . import Metric


class Accuracy(Metric):
    def __init__(self, pad_value: int = None):
        """
        用于词性标注
        :param pad_value: 被忽略的目标值
        """
        super(Accuracy, self).__init__(acc=0.)
        self._total = 0
        self._pad_value = pad_value
        self._total_correct = 0

    def step(self, y_pred: torch.Tensor, y: torch.Tensor):
        y_pred = torch.argmax(y_pred, dim=-1)

        if y.size() != y_pred.size():
            raise TypeError("y and y_pred should have the same shape")
        correct = torch.eq(y_pred, y)

        mask = torch.ones_like(y, dtype=torch.bool) if self._pad_value is None else y.ne(self._pad_value)
        self._total_correct += torch.sum(correct[mask]).item()
        self._total += mask.sum().item()

    def compute(self):
        return {'acc': (self._total_correct / self._total) if self._total != 0 else 0}

    def clear(self):
        self._total_correct = 0
        self._total = 0
