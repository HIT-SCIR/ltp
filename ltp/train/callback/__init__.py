#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from ltp.core import Registrable


class Callback(metaclass=Registrable):
    """
    回调函数基类
    """
    iteration: int
    epoch: int

    def __init__(self, iteration, epoch):
        self.iteration = iteration
        self.epoch = epoch

    def __call__(self, executor):
        self.call(executor)

    def init(self, executor):
        raise NotImplementedError()

    def call(self, executor):
        raise NotImplementedError()


from .csv_writer import CsvWriter
from .metric import MetricCallback
from .validation import ValidationCallback
from .tensorboard import TensorboardCallback

__all__ = ['Callback', 'CsvWriter', 'TensorboardCallback']
