#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import namedtuple

TaskInfo = namedtuple(
    "TaskInfo",
    [
        "task_name",
        "metric_name",
        "build_dataset",
        "validation_method"
    ]
)
