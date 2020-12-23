#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
__version__ = '4.1.2'

from . import const
from . import nn, utils
from . import data, optimization

from . import transformer_linear
from . import transformer_rel_linear
from . import transformer_biaffine
from . import transformer_multitask

try:
    from . import task_segmention
    from . import task_part_of_speech
    from . import task_named_entity_recognition
    from . import task_semantic_role_labeling
    from . import task_dependency_parsing
    from . import task_semantic_dependency_parsing
    from . import multitask
    from . import multitask_distill
except Exception as e:

    task_segmention = utils.LazyModule('task_segmention')
    task_part_of_speech = utils.LazyModule('task_part_of_speech')
    task_named_entity_recognition = utils.LazyModule('task_named_entity_recognition')
    task_semantic_role_labeling = utils.LazyModule('task_semantic_role_labeling')
    task_dependency_parsing = utils.LazyModule('task_dependency_parsing')
    task_semantic_dependency_parsing = utils.LazyModule('task_semantic_dependency_parsing')
    multitask = utils.LazyModule('multitask')
    multitask_distill = utils.LazyModule('multitask_distill')

from .frontend import LTP
