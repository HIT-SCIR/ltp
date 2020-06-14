#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from . import Callback


class ValidationCallback(Callback, alias="_valid"):
    def __init__(self, data_loader, metrics, task, device=None, dtype=None, non_blocking=False):
        super(ValidationCallback, self).__init__(None, 1)
        self.task = task
        self.preffix = '' if task == 'default' else (task + '_')
        self.data_loader = data_loader
        self.metrics = metrics
        self.device = device
        self.dtype = dtype
        self.non_blocking = non_blocking

    def init(self, executor):
        for metric in self.metrics:
            for name, value in metric.defaults.items():
                executor.trainer.state.add_attribute(self.preffix + name, value)
                # executor.add_progressbar_metric('dev_' + name)

    def call(self, executor):
        metrics = executor.evaluate_(self.data_loader, self.metrics, self.task)
        for metric in metrics:
            for attr, value in metric.compute().items():
                executor.trainer.state.update_attribute(self.preffix + attr, value)
            metric.clear()
