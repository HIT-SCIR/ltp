#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from . import Predictor
from ltp.utils import length_to_mask


class Biaffine(Predictor, alias='biaffine'):
    def __init__(self, fields, text_field='text', tag_field='tag'):
        super().__init__(fields)
        self.text_field = text_field
        self.tag_vocab = self.fields[tag_field].vocab.itos

    def _convert_idx_to_name(self, y, array_len):
        return [[self.tag_vocab[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]

    def predict(self, inputs, pred):
        srl_output, srl_length = pred
        mask = length_to_mask(srl_length)

        mask = mask.unsqueeze_(-1).expand(-1, -1, mask.size(1))
        mask = (mask & mask.transpose(-1, -2)).flatten(end_dim=1)
        index = mask[:, 0]
        mask = mask[index]

        srl_output = srl_output.flatten(end_dim=1)[index]
        srl_labels = torch.argmax(srl_output, dim=-1).cpu().numpy()
        srl_labels = self._convert_idx_to_name(srl_labels, mask.sum(dim=1))
        # srl_labels_res = []
        # for length in srl_length:
        #     srl_labels_res.append([])
        #     curr_srl_labels, srl_labels = srl_labels[:length], srl_labels[length:]
        #     srl_labels_res[-1].extend([get_entities(labels) for labels in curr_srl_labels])

        return srl_labels
