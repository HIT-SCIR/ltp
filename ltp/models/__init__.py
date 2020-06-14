#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
"""
模型在配置文件中的配置项为::
    [Model]
    class = "SrlTagging"

        [Model.init]
        label_num=77
        hidden_size=400
        pretrained="data/bert_wwm"
"""

from .model import Model
from .seq_tag_model import SequenceTaggingModel
from .multi_task_model import MultiTaskModel, SimpleMultiTaskModel
