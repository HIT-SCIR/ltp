#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from torch import nn
from ltp.core import Registrable


class Module(nn.Module, metaclass=Registrable):
    def __init__(self, input_size, label_num, dropout):
        super(Module, self).__init__()
        self.input_size = input_size
        self.label_num = label_num
        self.dropout = dropout

    def forward(self, embedding, seq_lens, gold=None):
        raise NotImplementedError()


from .linear import Linear
from .graph import Graph
from .biaffine_crf import BiaffineCRF
from .bilinear_crf import BiLinearCRF
from .relative_transformer import RelativeTransformer
