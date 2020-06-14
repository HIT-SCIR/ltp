#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from functools import partial
import torch
from torch import Tensor
from . import Trainer
from ltp.utils import clip_grad_norm, select_logits_with_mask
from ltp.predict import Predictor
from ltp.train.scheduler import TemperatureScheduler
from ltp.train.scheduler.temperature_scheduler import ConstantScheduler
from ltp.train.loss import Loss


class MultiTaskDistiller(Trainer, alias='multi_dist'):
    def __init__(self, config, temperature=8, kd_loss_weight=1, hard_label_weight=1):
        super().__init__(config)
        self.verbose = self.config.verbose
        self.tasks = self.config.tasks
        self.task = self.tasks['default']
        self.model = self.task.model

        self.temperature = temperature
        self._kd_loss_weight = kd_loss_weight
        self._hard_label_weight = hard_label_weight

        if hasattr(self.config, 'temperature_scheduler'):
            self.temperature_scheduler = TemperatureScheduler.from_params(self.config.temperature_scheduler)
        else:
            self.temperature_scheduler = ConstantScheduler()

        if hasattr(self.model, 'extractor'):
            self.data_extractor = lambda batch: self.model.extractor(batch)
        else:
            self.data_extractor = self.__data_extractor
        if self.verbose:
            print(self.model)

    @staticmethod
    def __data_extractor(batch, task):
        x = tuple(getattr(batch, name) for name, _ in task.fields if name in batch.input_fields)
        y = tuple(getattr(batch, name) for name, _ in task.fields if name in batch.target_fields)
        return x, y

    @property
    def kd_loss_weight(self) -> float:
        return self._kd_loss_weight * (1 - self.state.global_step / self.total_steps)

    @property
    def hard_label_weight(self) -> float:
        return self._hard_label_weight * (self.state.global_step / self.total_steps)

    def train(self, batch, task):
        self.task.optimizer.zero_grad()
        x, y = self.data_extractor(batch, self.tasks[task])

        if len(y) == 0:
            y = x
        if len(y) == 1:
            y = y[0]

        if isinstance(x, tuple):
            with torch.no_grad():
                logits_T = self.tasks[task].model(*x)
            logits = self.model[task](*x)
        elif isinstance(x, dict):
            with torch.no_grad():
                logits_T = self.tasks[task].model(**x)
            logits = self.model[task](**x)
        else:
            raise NotImplementedError("Model.extrator 的返回值 x 必须是一个tuple或者一个 dict")

        # 使用每个任务的 loss 计算 hard loss
        if isinstance(self.tasks[task].loss, Loss):
            # Hard Label Loss
            hard_label_loss = self.tasks[task].loss(logits, y)

            # Distill Loss
            temperature_calc = partial(self.temperature_scheduler, base_temperature=self.temperature)
            kd_loss = self.tasks[task].loss.distill(logits, logits_T, temperature_calc, self.task.loss, gold=y)
        elif isinstance(y, Tensor):
            # Hard Label Loss
            shape = logits.shape[-1]
            hard_label_loss = self.tasks[task].loss(logits.contiguous().view((-1, shape)), y.contiguous().view(-1))

            # Distill Loss
            mask = torch.ne(y, -1)
            logits = select_logits_with_mask(logits, mask)
            logits_T = select_logits_with_mask(logits_T, mask)

            temperature = self.temperature_scheduler(logits, logits_T, self.temperature)
            kd_loss = self.task.loss(logits, logits_T, temperature)
        else:
            raise NotImplementedError("多输出结果需要自行实现Loss")

        loss = kd_loss * self.kd_loss_weight + hard_label_loss * self.hard_label_weight
        loss.backward()

        clip_grad_norm(
            self.model.named_parameters() if self.config.pretrained_grad_norm else self.model.parameters(),
            self.config.grad_norm, norm_type=self.config.norm_type, pretrained_norm=self.config.pretrained_grad_norm
        )

        self.task.optimizer.step()
        if self.task.scheduler_type == 'step' and self.task.scheduler:
            self.task.scheduler.step()
        return loss.item()

    def after_train(self):
        if self.task.scheduler_type == 'epoch' and self.task.scheduler:
            self.task.scheduler.step()
        self.task.save(self.state)

    def before_eval(self, task):
        self.previous_training_flag = self.model.training
        self.model[task].eval()

    def eval(self, batch, task):
        x, y = self.data_extractor(batch, self.tasks[task])

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
        self.model[task].eval()
        if self.tasks[task].config.predictor:
            self.predictor = Predictor.from_params(
                self.tasks[task].config.predictor,
                fields=self.tasks[task].fields
            )
        else:
            print("No Predictor")

    def predict(self, batch, task):
        x, y = self.data_extractor(batch, self.tasks[task])

        if isinstance(x, tuple):
            y_pred = self.model(*x)
        elif isinstance(x, dict):
            y_pred = self.model(**x)
        else:
            raise NotImplementedError("Model.extrator 的返回值 x 必须是一个tuple或者一个 dict")

        return self.predictor(batch, y_pred)

    def after_predict(self, task):
        self.model.train(self.previous_training_flag)
