from .layze_import import LazyModule
from .layze_import import fake_import_pytorch_lightning

from .task_info import TaskInfo
from .length2mask import length_to_mask
from .convertor import map2device, convert2npy
from .deploy_model import deploy_model

try:
    from seqeval.metrics.sequence_labeling import get_entities
except Exception as e:
    from .get_entities import get_entities

try:
    from .common_train import common_train
except Exception as e:
    common_train = LazyModule('common_train', 'pytorch_lightning')
