#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss
from . import Metric


class CommonLoss(Metric, alias="Loss"):
    """
    Loss Metric: 用于计算LOSS

    :param loss_function: 损失函数
    :param item: 返回标量，而不是Tensor
    :param flat: 是否对预测结果进行平铺
    """

    def __init__(self, loss_function: Loss, item: bool = False, flat: bool = False):
        """
        Loss Metric: 用于计算LOSS

        :param loss_function: 损失函数
        :param item: 返回标量，而不是Tensor
        :param flat: 是否对预测结果进行平铺
        """

        super(CommonLoss, self).__init__(loss=float('inf'))
        self._total = 0
        self._flat = flat
        self._loss_sum = 0.
        self._return_item = item
        self.loss_function = loss_function

    def step(self, y_pred: Tensor, y: Tensor):
        if self._flat:
            shape = y_pred.shape[-1]
            y_pred, y = y_pred.contiguous().view((-1, shape)), y.contiguous().view(-1)
        loss = self.loss_function(y_pred, y)
        self._loss_sum += loss.item()
        self._total += 1

    def compute(self):
        if self._total == 0:
            raise ZeroDivisionError("Loss average is not computable.")
        return {'loss': (self._loss_sum / self._total) if self._total != 0 else 0}

    def clear(self):
        self._loss_sum = 0.
        self._total = 0
