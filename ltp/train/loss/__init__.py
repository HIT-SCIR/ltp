#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch import nn

from ltp.core import Registrable


class Loss(nn.modules.loss._Loss, metaclass=Registrable):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def forward(self, *input, **kwargs):
        raise NotImplementedError()

    def distill(self, inputs, targets, temperature_calc, distill_loss, gold=None):
        raise NotImplementedError()


Loss.weak_register("L1Loss", nn.L1Loss)
Loss.weak_register("L2Loss", nn.MSELoss)
Loss.weak_register("CTCLoss", nn.CTCLoss)
Loss.weak_register("NLLLoss", nn.NLLLoss)
Loss.weak_register("BCELoss", nn.BCELoss)
Loss.weak_register("MSELoss", nn.MSELoss)
Loss.weak_register("KLDivLoss", nn.KLDivLoss)
Loss.weak_register("NLLLoss2d", nn.NLLLoss2d)
Loss.weak_register("SmoothL1Loss", nn.SmoothL1Loss)
Loss.weak_register("PoissonNLLLoss", nn.PoissonNLLLoss)
Loss.weak_register("SoftMarginLoss", nn.SoftMarginLoss)
Loss.weak_register("CrossEntropyLoss", nn.CrossEntropyLoss)
Loss.weak_register("MarginRankingLoss", nn.MarginRankingLoss)
Loss.weak_register("BCEWithLogitsLoss", nn.BCEWithLogitsLoss)
Loss.weak_register("TripletMarginLoss", nn.TripletMarginLoss)
Loss.weak_register("HingeEmbeddingLoss", nn.HingeEmbeddingLoss)
Loss.weak_register("CosineEmbeddingLoss", nn.CosineEmbeddingLoss)
Loss.weak_register("MultiLabelMarginLoss", nn.MultiLabelMarginLoss)

from . import kd_loss
from . import task_loss

__all__ = ['Loss']
