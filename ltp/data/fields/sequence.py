#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import Counter, OrderedDict
from itertools import chain
from typing import Union

import torch

from torchtext.vocab import Vocab
from ltp.const import PAD
from . import Field
from ltp.data.dataset import Dataset


def dtype_to_attr(dtype):
    # convert torch.dtype to dtype string id
    # e.g. torch.int32 -> "int32"
    # used for serialization
    _, dtype = str(dtype).split('.')
    return dtype


class SequenceField(Field, alias='sequence'):
    """
    序列 Field，通常为target


    :param name: Field name
    :param bos: Begin Of Sentence，默认为空
    :param eos: End Of Sentence，默认为空
    :param unk: Unknown Tag, 默认为空
    :param pad: 默认为 [PAD] 或 -1
    :param dtype: torch.dtype，可以使用字符串
    :param pad_bias: 做值域变换，将pad的值变到1，默认开启
    :param preprocessing: 预处理
    :param postprocessing: 后处理
    :param max_length: 是否padding到最大长度，None即为不做特殊处理，默认为None
    :param include_lengths: 是否返回 length，默认为False
    :param use_vocab: 是否使用词表，默认为True
    :param is_target: 是否为target，默认为True
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

    def __init__(self, name, bos: Union[str, int] = None, eos: Union[str, int] = None,
                 unk: Union[str, int] = None, pad: Union[str, int] = None,
                 dtype=torch.long, pad_bias=True, preprocessing=None, postprocessing=None,
                 max_length: int = None, include_lengths=False, labels=None,
                 use_vocab=True, is_target: bool = True, **kwargs):

        super(SequenceField, self).__init__(name, preprocessing, postprocessing, is_target)

        self.unk = unk if (isinstance(unk, str) and use_vocab) or (isinstance(unk, int) and not use_vocab) else None
        self.bos = bos if (isinstance(bos, str) and use_vocab) or (isinstance(bos, int) and not use_vocab) else None
        self.eos = eos if (isinstance(eos, str) and use_vocab) or (isinstance(eos, int) and not use_vocab) else None
        if use_vocab:
            self.pad = pad if isinstance(pad, str) else PAD
        elif not use_vocab:
            self.pad = pad if isinstance(pad, int) else -1

        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype
        self.use_vocab = use_vocab
        self.max_length = max_length
        self.include_lengths = include_lengths

        self.pad_bias = pad_bias

        if labels:
            counter = Counter()
            counter.update(labels)
            specials = list(
                OrderedDict.fromkeys(
                    tok for tok in [self.unk, self.pad, self.bos, self.eos] + kwargs.pop('specials', []) if
                    tok is not None)
            )
            self.vocab = self.vocab_cls(counter, specials=specials)

    def build_vocab(self, *args, **kwargs):
        if hasattr(self, 'vocab'):
            return
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
                tok for tok in [self.unk, self.pad, self.bos, self.eos] + kwargs.pop('specials', []) if tok is not None)
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        return super(SequenceField, self).__setstate__(state)

    def __getstate__(self):
        attrs = super(SequenceField, self).__getstate__()
        attrs['dtype'] = dtype_to_attr(self.dtype)
        return attrs

    def pad_batch(self, minibatch: list):
        minibatch = list(minibatch)
        max_len = max(len(x) for x in minibatch)

        if self.max_length is not None:
            max_len = min(max_len, self.max_length + (self.bos, self.eos).count(None) - 2)

        padded, lengths = [], []

        for x in minibatch:
            padded.append(
                ([] if self.bos is None else [self.bos])
                + list(x[:max_len])
                + ([] if self.eos is None else [self.eos])
                + [self.pad] * max(0, max_len - len(x[:max_len]))
            )
            lengths.append(len(padded[-1]) - max(0, max_len - len(x[:max_len])))

        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        lengths = None
        if isinstance(arr, tuple):
            arr, lengths = arr

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
            arr = [[numericalization_func(x) for x in ex] for ex in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)
        var = var.contiguous()
        if self.include_lengths:
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
            return var, lengths
        return var

    def preprocess(self, x):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return x

    def process(self, batch, device=None):
        padded = self.pad_batch(batch)
        tensor = self.numericalize(padded, device=device)

        if not self.pad_bias:
            return tensor
        if isinstance(tensor, torch.Tensor):
            return tensor - 1
        else:
            tensor, length = tensor
            return tensor - 1, length
