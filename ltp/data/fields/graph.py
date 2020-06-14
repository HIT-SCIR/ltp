#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import Counter, OrderedDict
from itertools import chain
from typing import List

import torch

from torchtext.vocab import Vocab
from ltp.const import PAD
from . import Field
from ltp.data.dataset import Dataset


class GraphField(Field, alias='graph'):
    vocab_cls = Vocab

    def __init__(self, name, edge_spliter, pad: str = PAD, tag_spliter=None, use_vocab=True, **kwargs):
        """Graph Field

        Args:
            name(str): field name
            edge_spliter(str): 对边进行拆分的符号
            pad(str): 没有 tag 的地方使用 pad 填充
            tag_spliter(str): 对 tag 进行拆分的符号
            use_vocab(bool): 是否使用词典
            **kwargs:
        """
        super().__init__(name, **kwargs)
        self.edge_spliter = edge_spliter
        self.tag_spliter = tag_spliter
        self.use_vocab = (self.tag_spliter is not None) and use_vocab
        self.pad = pad
        if self.pad:
            self.pad_bias = True
        else:
            self.pad_bias = False

    def preprocess(self, arcs: List[str]):
        node_size = len(arcs)
        res = [arcs_info.split(self.edge_spliter) for arcs_info in arcs]
        if self.use_vocab:
            res = [[(index + 1, *arc.split(self.tag_spliter)) for arc in arcs] for index, arcs in enumerate(res)]
            res = [(int(src), int(target), label) for src, target, label in chain(*res)]
        else:
            res = [[(index + 1, int(arc)) for arc in arcs] for index, arcs in enumerate(res)]
        return node_size + 1, res

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
                tags = [arc[-1] for arc in x[1]]
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
        length, batched_edges = zip(*batch)
        max_edge_size = max(length)

        edge_tensor = torch.zeros(size=(len(length), max_edge_size, max_edge_size))
        tag_tensor = torch.zeros(size=(len(length), max_edge_size, max_edge_size), dtype=torch.long)

        for index, edges in enumerate(batched_edges):
            for edge in edges:
                edge_tensor[index, edge[0], edge[1]] = 1
                tag_tensor[index, edge[0], edge[1]] = self.vocab[edge[2]]

        return edge_tensor.to(device), tag_tensor.to(device) - 1
