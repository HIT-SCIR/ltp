#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from . import Predictor
from ltp.utils.seqeval import get_entities
from transformers import PreTrainedTokenizer


class Segment(Predictor, alias='segment'):
    def __init__(self, fields, text_field='text'):
        super().__init__(fields)
        self.text_field = text_field
        tokenizer = self.fields[text_field].tokenizer
        tokenizer: PreTrainedTokenizer

        self.id2text = tokenizer
        self.id2label = ['B-W', 'I-W']

    def convert_idx_to_name(self, y, array_len, id2label):
        return [[id2label[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]

    def predict(self, inputs, pred):
        input_text = getattr(inputs, self.text_field)
        target_len = input_text['text_length'].cpu().detach().numpy().tolist()

        pred = torch.argmax(pred, dim=-1).cpu().detach().numpy()
        text = input_text['input_ids'].cpu().detach().numpy()
        pred = self.convert_idx_to_name(pred, target_len, id2label=self.id2label)
        res = []
        for text, pred_single, length in zip(text, pred, target_len):
            text = [self.id2text.convert_ids_to_tokens(int(char)) for char in text[1:1 + length]]
            entities = get_entities(pred_single)
            sentence = ["".join(text[entity[1]:entity[2] + 1]) for entity in entities]
            res.append(sentence)
        return res
