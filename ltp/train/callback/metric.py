#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from . import Callback


class MetricCallback(Callback, alias="_metrics_wrapper"):
    def __init__(self, metrics):
        super(MetricCallback, self).__init__(1, None)
        self.metrics = metrics
        self.last_epoch = 0

    def init(self, executor):
        for metric in self.metrics:
            for name, value in metric.defaults.items():
                executor.trainer.state.add_attribute(name, value)
                executor.add_progressbar_metric(name)

    def call(self, executor):
        if self.last_epoch != executor.trainer.state.current_epoch:
            for metric in self.metrics:
                metric.clear()

        self.last_epoch = executor.trainer.state.current_epoch
        for metric in self.metrics:
            metric.step(executor.trainer.state.last_y_pred, executor.trainer.state.last_y)
            for attr, value in metric.compute().items():
                executor.trainer.state.update_attribute(attr, value)
