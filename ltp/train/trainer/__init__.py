#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from ltp.core import Registrable


class State(object):
    global_step = 0
    current_epoch = 0
    last_x = None
    last_y = None
    last_y_pred = None
    last_train_loss = float("inf")

    def get(self, attribute_name: str):
        return getattr(self, attribute_name)

    def load_state_dict(self, state):
        for k, v in state.__dict__.items():
            setattr(self, k, v)
        return self

    def add_attribute(self, attribute, value):
        if not hasattr(self, attribute):
            setattr(self, attribute, value)
        return self

    def update_attribute(self, attribute, value):
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        return self

    def reset(self):
        self.global_step = 0
        self.current_epoch = 0
        self.last_x = None
        self.last_y = None
        self.last_y_pred = None
        self.last_train_loss = float("inf")
        return self

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        return self.current_epoch == other.current_epoch \
               and self.global_step == other.global_step \
               and self.last_x == other.last_x \
               and self.last_y == other.last_y \
               and self.last_y_pred == other.last_y_pred \
               and self.last_train_loss == other.last_train_loss


class Trainer(metaclass=Registrable):
    state: State

    def __init__(self, config):
        self.state = State()
        self.config = config
        self.total_steps = 0
        self.predictor = None

    def init(self, total_steps):
        self.total_steps = total_steps

    def before_train(self):
        """
        每个EPOCH之前执行
        """
        pass

    def train(self, batch, task):
        raise NotImplementedError()

    def after_train(self):
        """
        每个EPOCH之后执行
        """
        pass

    def before_eval(self, task):
        """
        Eval 之前执行
        """
        pass

    def eval(self, batch, task):
        raise NotImplementedError()

    def after_eval(self, task):
        """
        Eval 之后执行
        """
        pass

    def before_predict(self, task):
        """
        预测之前执行
        """
        pass

    def predict(self, batch, task):
        raise NotImplementedError()

    def after_predict(self, task):
        """
        预测之后执行
        """
        pass


from .common_trainer import CommonTrainer
from .multi_task_distiller import MultiTaskDistiller

__all__ = ['Trainer', 'CommonTrainer', 'MultiTaskDistiller']
