#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import Counter, OrderedDict
from itertools import chain
import torch

from torchtext.vocab import Vocab

from . import Field
from ltp.data.dataset import Dataset
from ltp.const import UNK


class LabelField(Field, alias='label'):
    """
    可以用于文本分类等领域
    """
    vocab_cls = Vocab

    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype']

    def __init__(self, name, unk: str = UNK, preprocessing=None, postprocessing=None, use_vocab=True, dtype=torch.long,
                 is_target: bool = False):
        super(LabelField, self).__init__(name, preprocessing, postprocessing, is_target)

        self.unk = unk
        self.dtype = dtype
        self.use_vocab = use_vocab

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
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(
            OrderedDict.fromkeys(
                tok for tok in [self.unk] + kwargs.pop('specials', []) if tok is not None)
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    f"Specified Field dtype {self.dtype} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                )
            numericalization_func = self.dtypes[self.dtype]
            arr = [numericalization_func(x) for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)
        var = var.contiguous()
        return var

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        tensor = self.numericalize(batch, device=device)
        return tensor
