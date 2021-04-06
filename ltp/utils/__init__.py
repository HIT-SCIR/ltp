#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

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
    from .common_train import add_common_specific_args, add_tune_specific_args
except Exception as e:
    common_train = LazyModule('common_train', 'pytorch_lightning')


def dataset_cache_wrapper(extra=None, extra_builder=None):
    import os
    from transformers import AutoTokenizer
    from ltp.data.dataset import DatasetDict

    def process(func, data_dir, task_name, *args, **kwargs) -> DatasetDict:
        if 'model' in kwargs:
            model = kwargs['model']
            if 'tokenizer' not in kwargs:
                kwargs['tokenizer'] = AutoTokenizer.from_pretrained(model.hparams.transformer, use_fast=True)
            if 'max_length' not in kwargs:
                kwargs['max_length'] = model.transformer.config.max_position_embeddings
        # USER BUILD
        dataset: DatasetDict = func(data_dir, task_name, *args, **kwargs)
        return dataset

    def wrapper_maker(func):

        def func_wrapper(data_dir, task_name, *args, **kwargs) -> DatasetDict:
            cache_dir = os.path.join(data_dir, task_name)

            if 'model' in kwargs:
                model = kwargs['model']
                seed = model.hparams.seed
            else:
                seed = kwargs.get('seed', 19980524)
            seed = kwargs.get('seed', seed)

            if os.path.exists(cache_dir):
                try:
                    dataset = DatasetDict.load_from_disk(cache_dir)
                except Exception as e:
                    dataset = process(func, data_dir, task_name, *args, **kwargs)
                    dataset.save_to_disk(cache_dir)
            else:
                dataset = process(func, data_dir, task_name, *args, **kwargs)
                dataset.save_to_disk(cache_dir)

            dataset = dataset.shuffle(
                seed=seed,
                indices_cache_file_names={
                    k: d._get_cache_file_path(f"{task_name}-{k}-shuffled-index-{seed}") for k, d in
                    dataset.items()
                }
            )

            if extra_builder is not None:
                return dataset, extra_builder(dataset)

            return dataset, extra

        return func_wrapper

    return wrapper_maker
