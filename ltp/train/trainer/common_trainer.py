#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from torch import Tensor
from torchtext.data import Batch

from . import Trainer
from ltp.utils import clip_grad_norm
from ltp.predict import Predictor
from ltp.train.loss import Loss


class CommonTrainer(Trainer, alias='default'):
    def __init__(self, config):
        super().__init__(config)
        self.verbose = self.config.verbose
        self.task = self.config.tasks['default']
        self.model = self.task.model

        if hasattr(self.model, 'extractor'):
            self.data_extractor = lambda batch: self.model.extractor(batch)
        else:
            self.data_extractor = self.__data_extractor
        if self.verbose:
            print(self.model)

    def __data_extractor(self, batch: Batch):
        x = tuple(getattr(batch, name) for name, _ in self.task.fields if name in batch.input_fields)
        y = tuple(getattr(batch, name) for name, _ in self.task.fields if name in batch.target_fields)
        return x, y

    def before_train(self):
        self.model.train()

    def train(self, batch, task):
        self.task.optimizer.zero_grad()
        x, y = self.data_extractor(batch)

        if len(y) == 0:
            y = x
        if len(y) == 1:
            y = y[0]

        if isinstance(x, tuple):
            y_pred = self.model(*x, gold=y)
        elif isinstance(x, dict):
            y_pred = self.model(**x, gold=y)
        else:
            raise NotImplementedError("Model.extrator 的返回值 x 必须是一个tuple或者一个 dict")

        if isinstance(self.task.loss, Loss):
            loss = self.task.loss(y_pred, y)
        elif isinstance(y, Tensor):
            shape = y_pred.shape[-1]
            loss = self.task.loss(y_pred.contiguous().view((-1, shape)), y.contiguous().view(-1))
        else:
            raise NotImplementedError("多输出结果需要自行实现Loss")
        loss_item = loss.item()
        loss.backward()

        clip_grad_norm(
            self.model.named_parameters() if self.config.pretrained_grad_norm else self.model.parameters(),
            self.config.grad_norm, norm_type=self.config.norm_type, pretrained_norm=self.config.pretrained_grad_norm
        )
        self.task.optimizer.step()
        if self.task.scheduler_type == 'step' and self.task.scheduler:
            self.task.scheduler.step()
        return loss_item

    def after_train(self):
        if self.task.scheduler_type == 'epoch' and self.task.scheduler:
            self.task.scheduler.step()
        self.task.save(self.state)

    def before_eval(self, task):
        self.previous_training_flag = self.model.training
        self.model.eval()

    def eval(self, batch, task):
        self.model.eval()
        x, y = self.data_extractor(batch)

        if len(y) == 0:
            y = x
        if len(y) == 1:
            y = y[0]

        if isinstance(x, tuple):
            y_pred = self.model(*x)
        elif isinstance(x, dict):
            y_pred = self.model(**x)
        else:
            raise NotImplementedError("Model.extrator 的返回值 x 必须是一个tuple或者一个 dict")

        return x, y_pred, y

    def after_eval(self, task):
        self.model.train(self.previous_training_flag)

    def before_predict(self, task):
        self.previous_training_flag = self.model.training
        self.model.eval()
        if self.task.config.predictor:
            self.predictor = Predictor.from_params(
                self.task.config.predictor,
                fields=self.task.fields
            )
        else:
            print("No Predictor")

    def predict(self, batch, task):
        self.model.eval()
        x, y = self.data_extractor(batch)

        if isinstance(x, tuple):
            y_pred = self.model(*x)
        elif isinstance(x, dict):
            y_pred = self.model(**x)
        else:
            raise NotImplementedError("Model.extrator 的返回值 x 必须是一个tuple或者一个 dict")

        return self.predictor(batch, y_pred)

    def after_predict(self, task):
        self.model.train(self.previous_training_flag)
