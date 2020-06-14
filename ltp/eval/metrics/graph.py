#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Tuple

import torch
from . import Metric
from ltp.utils import length_to_mask


class GraphMetrics(Metric, alias='graph'):
    """
    Graph Parser Metric(f1)
    """

    def __init__(self):
        """
        Graph Parser Metric(f1)
        """
        super(GraphMetrics, self).__init__(f1=0., p=0., r=0.)
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0

    def get_entities(self, arcs, labels):
        arcs = torch.nonzero(arcs).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        res = []
        for arc in arcs:
            arc = tuple(arc)
            label = labels[arc]
            res.append((*arc, label))

        return set(res)

    def step(self, y_pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]):
        arc_pred, label_pred, seq_len = y_pred
        arc_real, label_real = y

        arc_real = arc_real > 0.5
        arc_pred = torch.sigmoid(arc_pred) > 0.5
        # to [B, L+1, L+1]
        label_pred = torch.argmax(label_pred, dim=-1)

        mask = length_to_mask(seq_len + 1)
        mask[:, 0] = False  # ignore the first token of each sentence
        mask = mask.unsqueeze(-1).expand_as(arc_pred)

        arc_pred[mask == False] = False

        true_entities = self.get_entities(arc_real, label_real)
        pred_entities = self.get_entities(arc_pred, label_pred)

        self.nb_correct += len(true_entities & pred_entities)
        self.nb_pred += len(pred_entities)
        self.nb_true += len(true_entities)

    @property
    def precision(self):
        return self.nb_correct / self.nb_pred if self.nb_pred > 0 else 0

    @property
    def recall(self):
        return self.nb_correct / self.nb_true if self.nb_true > 0 else 0

    @property
    def f1beta(self):
        p = self.precision
        r = self.recall
        score = 2 * p * r / (p + r) if (p + r > 0) else 0
        return score

    def compute(self):
        return {'f1': self.f1beta, 'p': self.precision, 'r': self.recall}

    def clear(self):
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0
