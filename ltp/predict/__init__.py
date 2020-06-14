#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
"""
预测器，在配置文件中需配置此项::

    [Predictor]
    class = "simple"
"""

from typing import Dict, Any
from ltp.core import Registrable


class Predictor(metaclass=Registrable):
    """
    若需要使用Predict功能，则需要配置此项::

        [Predictor]
        class = "simple"
            [Predictor.init]
            len_field = 'text_len'

    """
    fields: Dict[str, Any]

    def __init__(self, fields):
        self.fields = dict(fields)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, inputs, pred):
        """
        对结果进行预测
        """
        raise NotImplementedError("Predictor 需要实现predict函数")

    @property
    def input_fields(self):
        return [field for field in self.fields.values() if not field.is_target]

    @property
    def target_fields(self):
        return [field for field in self.fields.values() if field.is_target]


from .simple import Simple
from .segment import Segment
from .biaffine import Biaffine

__all__ = ['Simple', 'Segment', 'Biaffine']
