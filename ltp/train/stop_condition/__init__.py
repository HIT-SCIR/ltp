#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from .early_stopping import EarlyStopping


class NoStopping(object):
    """
    永不停止
    """
    def __call__(self, state):
        return False


__all__ = ['NoStopping', 'EarlyStopping']
