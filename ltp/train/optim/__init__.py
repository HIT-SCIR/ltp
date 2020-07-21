#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import inspect
from torch import optim
from ltp.core import Registrable


class Optimizer(optim.Optimizer, metaclass=Registrable):
    @classmethod
    def from_extra(cls, extra: dict, subcls=None):
        if subcls is None:
            return {'params': filter(lambda p: p.requires_grad, extra['model'].parameters())}
        sig = inspect.signature(subcls)
        params = sig.parameters.keys()
        if 'params' in params:
            return {'params': filter(lambda p: p.requires_grad, extra['model'].parameters())}
        elif 'named_params' in params:
            named_params = []
            for (n, p) in extra['model'].named_parameters():
                if p.requires_grad:
                    named_params.append((n, p))
            return {'named_params': named_params}
        else:
            return {}


Optimizer.weak_register("Adadelta", optim.Adadelta)
Optimizer.weak_register("Adagrad", optim.Adagrad)
Optimizer.weak_register("Adam", optim.Adam)
Optimizer.weak_register("AdamW", optim.AdamW)
Optimizer.weak_register("SparseAdam", optim.SparseAdam)
Optimizer.weak_register("Adamax", optim.Adamax)
Optimizer.weak_register("ASGD", optim.ASGD)
Optimizer.weak_register("SGD", optim.SGD)
Optimizer.weak_register("Rprop", optim.Rprop)
Optimizer.weak_register("RMSprop", optim.RMSprop)
Optimizer.weak_register("LBFGS", optim.LBFGS)

try:
    import torch_optimizer

    Optimizer.weak_register("AccSGD", torch_optimizer.AccSGD)
    Optimizer.weak_register("AdaBound", torch_optimizer.AdaBound)
    Optimizer.weak_register("AdaMod", torch_optimizer.AdaMod)
    Optimizer.weak_register("DiffGrad", torch_optimizer.DiffGrad)
    Optimizer.weak_register("Lamb", torch_optimizer.Lamb)
    Optimizer.weak_register("NovoGrad", torch_optimizer.NovoGrad)
    Optimizer.weak_register("PID", torch_optimizer.PID)
    Optimizer.weak_register("QHM", torch_optimizer.QHM)
    Optimizer.weak_register("RAdam", torch_optimizer.RAdam)
    Optimizer.weak_register("SGDW", torch_optimizer.SGDW)
    Optimizer.weak_register("Yogi", torch_optimizer.Yogi)
except Exception as e:
    pass

from .pretrained_optim import PretrainedOptim, BertAdamW
from .task_optim import BertAdamW4CRF

__all__ = ['Optimizer', 'PretrainedOptim', 'BertAdamW', 'BertAdamW4CRF']
