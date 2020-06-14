#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from . import Predictor


class Simple(Predictor, alias='simple'):
    """
    简单预测器

    :param fields: Tuple(str, Field) 数据的 Field 对象，配置文件不需要填写
    :param text_field: str 提供文本以及长度的 Field，默认是第一个 Filed
    :param target_filed: str 提供解码词典的 Field, 默认是第一个 Target Field (提供他的词表)
    """

    def __init__(self, fields, text_field: str = 'text', target_filed: str = None):
        """
        简单预测器
        """
        super(Simple, self).__init__(fields)
        self.text_field = self.fields[text_field]
        if target_filed is None:
            label_field = self.target_fields[0]
        else:
            label_field = self.fields.get(target_filed, default=self.target_fields[0])

        if label_field.use_vocab:
            self.itos = label_field.vocab.itos[1:]

    def convert_idx_to_name(self, y, array_len, id2label):
        if id2label:
            return [[id2label[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]
        else:
            return [[idx for idx in row[:row_len]] for row, row_len in zip(y, array_len)]

    def predict(self, inputs, pred):
        text_field = getattr(inputs, self.text_field)
        target_len = text_field[1] if isinstance(text_field, tuple) else text_field
        target_len = target_len.cpu().detach().numpy().tolist()

        pred = torch.argmax(pred, dim=-1).cpu().detach().numpy()
        pred = self.convert_idx_to_name(pred, target_len, self.itos)
        return pred
