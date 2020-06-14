#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch.optim.lr_scheduler import LambdaLR
from . import Scheduler


def StepExponentialLR(optimizer, decay_rate, decay_steps, staircase=True, last_epoch=-1):
    def lr_lambda(current_step):
        if staircase:
            return decay_rate ** (current_step // decay_steps)
        return decay_rate ** (current_step / decay_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


Scheduler.weak_register("StepExponentialLR", StepExponentialLR)
