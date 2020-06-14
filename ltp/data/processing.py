#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from itertools import chain
from typing import List, Union

from ltp.core import Registrable
from ltp.data import Vocab


class Processing(object):
    """预/后处理基类"""

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError()


class PreProcessing(Processing, metaclass=Registrable):
    """预处理基类"""

    def call(self, x: List[str]):
        raise NotImplementedError()


class PostProcessing(Processing, metaclass=Registrable):
    """后处理基类"""

    def call(self, x, vocab: Union[Vocab, None]):
        raise NotImplementedError()


class BioEncoder(PreProcessing):
    """将词序列转换成BIO编码"""

    def call(self, x: List[str]):
        return list(chain(*(["B"] + ["I"] * (len(word) - 1) for word in x)))
