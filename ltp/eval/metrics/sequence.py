#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from . import Metric
import numpy as np
from ltp.utils.seqeval import get_entities


class Sequence(Metric):
    """
    用于命名实体识别或其他Span序列任务
    """

    def __init__(self, id2label, pad_value=-1, suffix=False, no_suffix=False):
        super(Sequence, self).__init__(f1=0., p=0., r=0.)
        self.field = id2label
        self.pad_value = pad_value
        self.suffix = suffix
        self.no_suffix = no_suffix

        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0
        self._id2label = id2label if isinstance(id2label, list) else None

    @property
    def id2label(self):
        if not self._id2label:
            # Lazy Import
            self._id2label = self.field.vocab.itos[1:]
            if self.no_suffix:
                self._id2label = [tag + '-W' for tag in self.field.vocab.itos[1:]]
            else:
                self._id2label = self.field.vocab.itos[1:]
        return self._id2label

    def convert_idx_to_name(self, y, array_indexes):
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_pred, y_true

    def step(self, y_pred: torch.Tensor, y: torch.Tensor):
        y_pred = torch.argmax(y_pred, dim=-1)
        y_pred, y = self.predict(y_pred, y)

        true_entities = set(get_entities(y, self.suffix))
        pred_entities = set(get_entities(y_pred, self.suffix))

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

    @classmethod
    def from_extra(cls, extra: dict):
        init = extra["config"]['init']
        id2label = init['id2label']

        for field_name, field in extra['fields']:
            if field_name == id2label:
                return {'id2label': field}

        return {'id2label': None}
