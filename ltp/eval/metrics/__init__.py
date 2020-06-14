#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from typing import Dict

from ...core import Registrable


class Metric(metaclass=Registrable):
    """
    Metric 基础类，在配置文件中的配置项为::

        [[Metrics]]
        class = "Accuracy"
        [Metrics.init]
        pad_value = -1
    """
    eps = 1e-6
    defaults: Dict[str, float]

    def __init__(self, **defaults):
        """

        """
        self.defaults = defaults

    def step(self, y_pred, y):
        """
        步进一次
        :param y_pred: 预测值
        :param y: 真实值
        """
        raise NotImplementedError()

    def compute(self) -> Dict[str, float]:
        """
        返回 metrics 结果
        :return: Dict[str, float]
        """
        raise NotImplementedError()

    def clear(self):
        """
        把已经统计的Metrics结果清0
        """
        raise NotImplementedError()


from .accuracy import Accuracy
from .common_loss import CommonLoss
from .tree import TreeMetrics
from .graph import GraphMetrics
from .sequence import Sequence
from .biaffine_crf_span import BiaffineCRFSpan
from .segment import Segment

__all__ = [
    'Metric', 'Accuracy', 'CommonLoss', 'TreeMetrics', 'GraphMetrics',
    'Sequence', 'BiaffineCRFSpan'
]
