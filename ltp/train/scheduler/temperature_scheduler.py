#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from . import TemperatureScheduler


class ConstantScheduler(TemperatureScheduler, alias='constant'):
    def forward(self, logits_S, logits_T, base_temperature):
        return base_temperature


class FlswScheduler(TemperatureScheduler, alias='flsw'):
    def __init__(self, beta=1, gamma=1, eps=1e-4):
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)
            t = t / (torch.norm(t, dim=-1, keepdim=True) + self.eps)
            w = torch.pow((1 - (v * t).sum(dim=-1)), self.gamma)
            tau = base_temperature + (w.mean() - w) * self.beta
        return tau


class CwsmScheduler(TemperatureScheduler, alias='cwsm'):
    def __init__(self, beta=1):
        self.beta = beta

    def forward(self, logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        with torch.no_grad():
            v = torch.softmax(v, dim=-1)
            v_max = v.max(dim=-1)[0]
            w = 1 / (v_max + 1e-3)
            tau = base_temperature + (w.mean() - w) * self.beta
        return tau
