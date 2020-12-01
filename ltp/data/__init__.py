#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from . import utils

try:
    from . import dataset
except Exception as e:
    from types import ModuleType


    class _Dataset(ModuleType):
        def __init__(self):
            super().__init__("dataset")

        def __getattr__(self, item):
            print("Need Install datasets!!!")
            print("pip install datasets")


    dataset = _Dataset()
