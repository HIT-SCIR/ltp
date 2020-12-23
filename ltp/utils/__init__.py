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
    from .common_train import common_train, tune_train
except Exception as e:
    common_train = LazyModule('common_train', 'pytorch_lightning')


def dataset_cache_wrapper(extra=None, extra_builder=None):
    import os
    from ltp.data.dataset import DatasetDict

    def wrapper_maker(func):

        def func_wrapper(model, data_dir, task_name):
            cache_dir = os.path.join(data_dir, task_name)
            if os.path.exists(cache_dir):
                dataset = DatasetDict.load_from_disk(cache_dir)
            else:
                dataset: DatasetDict = func(model, data_dir, task_name)
                dataset.save_to_disk(cache_dir)

            if extra_builder is not None:
                return dataset, extra_builder(dataset)

            return dataset, extra

        return func_wrapper

    return wrapper_maker
