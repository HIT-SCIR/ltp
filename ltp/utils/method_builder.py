#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import types
import torch
import datasets
import torch.utils.data

from ltp import optimization
from ltp.data.utils import collate
from ltp.utils.task_info import TaskInfo
from ltp.nn import BaseModule as Model


def default_build_method(model: Model, task_info: TaskInfo):
    dataset, metric = task_info.build_dataset(
        data_dir=model.hparams.data_dir,
        task_name=task_info.task_name,
        model=model
    )

    def train_dataloader(self: Model):
        res = torch.utils.data.DataLoader(
            dataset[datasets.Split.TRAIN],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return res

    def training_step(self: Model, batch, batch_nb):
        result = self.forward(**batch)
        self.log("loss", result.loss.item())
        return result.loss

    def val_dataloader(self: Model):
        return torch.utils.data.DataLoader(
            dataset[datasets.Split.VALIDATION],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self: Model):
        return torch.utils.data.DataLoader(
            dataset[datasets.Split.TEST],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    # AdamW + LR scheduler
    def configure_optimizers(self: Model):
        num_epoch_steps = (len(dataset[datasets.Split.TRAIN]) + self.hparams.batch_size - 1) // self.hparams.batch_size
        num_train_steps = num_epoch_steps * self.hparams.max_epochs
        optimizer, scheduler = optimization.from_argparse_args(
            self.hparams,
            model=self,
            num_train_steps=num_train_steps,
            n_transformer_layers=self.transformer.config.num_hidden_layers,
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    model.configure_optimizers = types.MethodType(configure_optimizers, model)

    model.train_dataloader = types.MethodType(train_dataloader, model)
    model.training_step = types.MethodType(training_step, model)

    validation_step, validation_epoch_end = task_info.validation_method(
        metric, task=task_info.task_name, preffix='val'
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = task_info.validation_method(
        metric, task=task_info.task_name, preffix='test'
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)
