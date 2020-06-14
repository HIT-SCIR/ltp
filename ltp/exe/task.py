#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import functools
import os
from torchtext.data import Batch


class TaskConfig(object):
    """
    和具体Trainer相关的配置文件
    """

    def __init__(self, config, global_config):
        self.epoch = global_config.epoch
        self.model = config.get("Model")
        self.ckptcfg = config.get("Checkpoint")
        self.fields = config.get("Fields", [])
        self.dataset = config.get("Dataset", None)
        self.loss = config.get("Loss", None)
        self.predictor = config.get("Predictor", None)
        self.metrics = config.get("Metrics", [])
        self.optimizer = config.get("Optimizer")
        self.scheduler = config.get("Scheduler", None)


class Task(object):
    """模型的训练任务
    """

    def __init__(self, config: TaskConfig, device):
        from ltp.models import Model
        from ltp.data import Field
        from ltp.eval import Metric
        from ltp.train import Loss, CheckpointManager, Optimizer

        self.device = device
        self.config = config
        self.model = Model.from_params(config.model)
        self.model.to(self.device)

        self.fields = []

        for field_config in config.fields:
            if not len(field_config):
                continue
            if "name" not in field_config['init']:
                name = field_config["name"]
                field_config.setdefault("name", default=name)
            field = Field.from_params(field_config)
            self.fields.append((field.name, field))

        self.loss = Loss.from_params(config.loss)
        self.metrics = [Metric.from_params(metric, extra={'fields': self.fields})
                        for metric in self.config.metrics if len(metric)]

        self.optimizer = Optimizer.from_params(self.config.optimizer, extra={'model': self.model})

        self.scheduler = None
        self.scheduler_type = None

        self.ckptmgr = CheckpointManager(**config.ckptcfg)

    def build_scheduler(self, num_training_steps):
        from ltp.train import Scheduler
        if self.config.scheduler:
            self.scheduler_type = self.config.scheduler.get('type', 'epoch')
            self.scheduler = Scheduler.from_params(self.config.scheduler, self.optimizer, extra={
                'num_training_steps': num_training_steps
            })
        else:
            self.scheduler_type = None
            self.scheduler = None

    def train_dataset(self, batch_size, shuffle=True):
        from ltp.data import Dataset, DataLoader

        path = self.config.dataset['path']
        train = Dataset.from_params(
            self.config.dataset,
            path=path,
            file=self.config.dataset['train'],
            fields=self.fields
        )

        for name, field in self.fields:
            if not hasattr(field, 'use_vocab') or not field.use_vocab:
                continue
            if not hasattr(field, 'vocab'):
                field.build_vocab(train)

        if 'validation' in self.config.dataset:
            valid = Dataset.from_params(
                self.config.dataset,
                path=path,
                file=self.config.dataset['validation'],
                fields=self.fields
            )
        else:
            ratio = self.config.dataset.get('ratio', 0.9)
            train, valid = train.split(ratio)

        def dataloader(dataset, class_):
            collate_fn = functools.partial(Batch, dataset=dataset, device=self.device)
            dataloader_func = functools.partial(class_, shuffle=shuffle,
                                                batch_size=batch_size,
                                                collate_fn=collate_fn)
            return dataloader_func(dataset)

        train_dataloader = dataloader(train, DataLoader)
        valid_dataloader = dataloader(valid, DataLoader)
        return train_dataloader, valid_dataloader

    def load_dataset(self, file=None, batch_size=10, shuffle=False):
        from ltp.data import Dataset
        from torch.utils.data import DataLoader

        path = file or self.config.dataset['path']
        test = Dataset.from_params(
            self.config.dataset,
            path=path,
            file=self.config.dataset['test'],
            fields=self.fields
        )
        collate_fn = functools.partial(Batch, dataset=test, device=self.device)
        dataloader = functools.partial(DataLoader, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
        return dataloader(test)

    def restore(self, state=None):
        self.ckptmgr['restore'].load(self, state)

    def load(self, state=None, path=None):
        if path:
            self.ckptmgr[path].load(self, state)
        else:
            self.ckptmgr['best'].load(self, state)

    def save(self, state):
        self.ckptmgr.save(self, state)
