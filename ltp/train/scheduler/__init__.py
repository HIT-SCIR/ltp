#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import inspect

from ltp.core import Registrable
from torch.optim import lr_scheduler
from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


class Scheduler(lr_scheduler._LRScheduler, metaclass=Registrable):
    """
    调度器
    """

    @classmethod
    def from_extra(cls, extra: dict, subcls=None):
        sig = inspect.signature(subcls)
        params = sig.parameters.keys()

        res = {}

        config = extra['config']
        init = config.get('init', {})
        if ('num_training_steps' in params) and ('num_training_steps' not in init):
            res['num_training_steps'] = extra['num_training_steps']

        if ('num_warmup_steps' in params) and ('num_warmup_steps' not in init) and ('warmup_proportion' in config):
            res['num_warmup_steps'] = extra['num_training_steps'] * config['warmup_proportion']
        return res


Scheduler.weak_register("LambdaLR", lr_scheduler.LambdaLR)
Scheduler.weak_register("CosineAnnealingLR", lr_scheduler.CosineAnnealingLR)
Scheduler.weak_register("ExponentialLR", lr_scheduler.ExponentialLR)
Scheduler.weak_register("MultiStepLR", lr_scheduler.MultiStepLR)
Scheduler.weak_register("ReduceLROnPlateau", lr_scheduler.ReduceLROnPlateau)
Scheduler.weak_register("StepLR", lr_scheduler.StepLR)

try:
    Scheduler.weak_register("CosineAnnealingWarmRestarts", lr_scheduler.CosineAnnealingWarmRestarts)
    Scheduler.weak_register("CyclicLR", lr_scheduler.CyclicLR)
except Exception as e:
    pass

Scheduler.weak_register("ConstantLR", get_constant_schedule)
Scheduler.weak_register("ConstantLRW", get_constant_schedule_with_warmup)
Scheduler.weak_register("CosineLRW", get_cosine_schedule_with_warmup)
Scheduler.weak_register("CosineHrLRW", get_cosine_with_hard_restarts_schedule_with_warmup)
Scheduler.weak_register("LinearLRW", get_linear_schedule_with_warmup)

from . import scheduler


class TemperatureScheduler(metaclass=Registrable):
    """
    蒸馏温度调度器
    """

    def __call__(self, logits_S, logits_T, base_temperature):
        return self.forward(logits_S, logits_T, base_temperature)

    def forward(self, logits_S, logits_T, base_temperature):
        raise NotImplementedError("未实现")


from . import temperature_scheduler

__all__ = ['Scheduler', 'TemperatureScheduler']
