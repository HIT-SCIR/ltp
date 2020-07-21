#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from typing import Dict, Generic, TypeVar

import torch
from ltp.core import Registrable
from ltp.data import PreProcessing, PostProcessing


class Field(metaclass=Registrable):
    """ 通用基础 Field 类型

    每个数据集都会包含各种类型的数据，它们都可以被表示为Field，它描述了数据被处理的过程。

    在配置文件中配置项为::

        [[Fields]]
        class = "BertField"
        name = "text"
        [Fields.init]
        batch_first=true
        pretrained="data/albert_g/vocab.txt"
        config={"do_lower_case" = false}
        include_lengths=false

        [[Fields]]
        class = "TagField"
        name = "text_len"
        [Fields.init]
        is_target=false
        use_vocab=false

    属性:
        is_target: 是否是目标域。
            Default: False
    """

    ignore = []

    def __init__(self, name, preprocessing=None, postprocessing=None, is_target=False):
        self.name = name
        self.is_target = is_target
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def preprocess(self, x):
        raise NotImplementedError()

    def process(self, batch, device=None, *args, **kwargs):
        raise NotImplementedError()

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self.ignore}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        return self.__getstate__()

    def load_state_dict(self, state):
        self.__setstate__(state)

    @classmethod
    def from_extra(cls, extra: dict, subcls=None):
        res = {}
        init = extra["config"]['init']
        if "preprocessing" in init:
            res["preprocessing"] = PreProcessing.from_params(init["preprocessing"])
        if "postprocessing" in init:
            res["postprocessing"] = PostProcessing.from_params(init["postprocessing"])
        return res


from .text import TextField, MixedTextField
from .label import LabelField
from .sequence import SequenceField
from .graph import GraphField
from .biaffine import BiaffineField

__all__ = ['Field', 'TextField', 'MixedTextField', 'LabelField', 'SequenceField', 'GraphField', 'BiaffineField']
