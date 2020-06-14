#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import Counter, OrderedDict
from itertools import chain
from typing import List

import torch

from torchtext.vocab import Vocab
from ltp.data.dataset import Dataset

from . import Field


class BiaffineField(Field, alias='biaffine'):
    """
    Biaffine 域

    Args:
        name: label name
        use_vocab: 是否使用词典
        pad: 无 label 的 label 位置使用 pad 填充
        labels: 可以给定 label 而不是通过统计数据得到
    """
    vocab_cls = Vocab

    def __init__(self, name, use_vocab=True, pad='O', labels=None, **kwargs):
        super().__init__(name, **kwargs)
        self.pad = pad
        self.use_vocab = use_vocab
        if labels:
            counter = Counter()
            counter.update(labels)
            specials = list(
                OrderedDict.fromkeys(
                    tok for tok in [self.pad] + kwargs.pop('specials', []) if tok is not None)
            )
            self.vocab = self.vocab_cls(counter, specials=specials)

    def preprocess(self, inputs: List[List[str]]):
        heads, *labels = inputs
        sentences_length = len(heads)
        heads = [i for i, head in enumerate(heads) if head == 'Y']

        target_labels = []
        for predicate, label in zip(heads, labels):
            label = [(predicate, index, label) for index, label in enumerate(label)]
            target_labels.extend(label)

        return sentences_length, heads, target_labels

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                tags = [label[-1] for label in x[-1]]
                try:
                    counter.update(tags)
                except TypeError:
                    print("error")
                    counter.update(chain.from_iterable(tags))
        specials = list(
            OrderedDict.fromkeys(
                tok for tok in [self.pad] + kwargs.pop('specials', []) if tok is not None)
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def process(self, batch, device=None, *args, **kwargs):
        sentence_lengths, heads, labels = zip(*batch)
        label_set = set()

        batch_size = len(batch)
        max_sentences_length = max(sentence_lengths)

        label_tensor = torch.zeros([batch_size] + [max_sentences_length] * 2, dtype=torch.long)
        # (sent_idx, predicate, word_index) => span_label
        for sent_idx, sent_target_labels in enumerate(labels):
            for predicate, index, label in sent_target_labels:
                label_index = self.vocab.stoi[label]
                if label_index == 0:
                    continue
                label_tensor[sent_idx, predicate, index] = label_index
                label_set.add((sent_idx, predicate, index, label_index))
        return label_tensor.to(device), label_set
