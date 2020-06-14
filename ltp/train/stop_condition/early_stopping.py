#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Union, List

default_float_rounding = 4


class EarlyStopping(object):
    """
    早停止
    """
    metric: Union[str, List[str]]
    patience: int
    float_rounding: int

    def __init__(self, patience: int = None, metric: Union[str, List[str]] = None, float_rounding: int = None):
        if patience is None:
            patience = 5
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        self.counter = 0
        self.best_score = None
        self.metric = 'loss' if metric is None else metric
        self.patience = patience
        self.float_rounding = default_float_rounding if float_rounding is None else float_rounding

    def better(self, metric):
        return round(metric, self.float_rounding) >= round(self.best_score, self.float_rounding)

    def __call__(self, state):
        if isinstance(self.metric, list):
            metric = sum([state.get(metric_name) for metric_name in self.metric])
        else:
            metric = state.get(self.metric)

        if self.best_score is None:
            self.best_score = metric
        elif self.better(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
