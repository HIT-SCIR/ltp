#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from typing import Tuple, Any
from itertools import chain

import torch
from . import Metric
from ltp.utils import length_to_mask
from ltp.utils.seqeval import get_entities


class BiaffineCRFSpan(Metric, alias='biaffine_crf_span'):
    """
    用于 语义角色标注
    """

    def __init__(self, id2label):
        super(BiaffineCRFSpan, self).__init__(f1=0., p=0., r=0.)
        self.field = id2label
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0

    def get_entities(self, labels):
        labels = labels.cpu().detach().numpy()
        labels = [self.field.vocab.itos[label] for label in labels]
        labels = get_entities(labels)

        return set(labels)

    def get_entities_with_list(self, labels):
        labels = [self.field.vocab.itos[label] for label in chain(*labels)]
        labels = get_entities(labels)
        return set(labels)

    def step(self, y_pred: Tuple[torch.Tensor, torch.Tensor, Any], y: Tuple[torch.Tensor, set]):
        rel_gold, rels_gold_set = y
        rels_scores, seq_lens, crf = y_pred

        mask = length_to_mask(seq_lens)
        mask = mask.unsqueeze_(-1).expand(-1, -1, mask.size(1))
        mask = mask & mask.transpose(-1, -2)
        mask = mask.flatten(end_dim=1)
        index = mask[:, 0]

        rel_gold = rel_gold.flatten(end_dim=1)[index]

        mask = mask[index]
        pred_entities = crf.decode(rels_scores.flatten(end_dim=1)[index], mask)

        rel_entities = self.get_entities(rel_gold[mask])
        pred_entities = self.get_entities_with_list(pred_entities)

        self.nb_correct += len(rel_entities & pred_entities)
        self.nb_pred += len(pred_entities)
        self.nb_true += len(rel_entities)

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
