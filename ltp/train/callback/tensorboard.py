#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os
from . import Callback

default_main_tag = "tensorboard"
default_save_directory = './log'


class TensorboardCallback(Callback, alias="tensorboard"):
    def __init__(self, iteration, epoch, save_directory=default_save_directory, main_tag=default_main_tag,
                 data_extractor=lambda state: {"loss": state.last_train_loss}):
        super(TensorboardCallback, self).__init__(iteration, epoch)
        from torch.utils.tensorboard import SummaryWriter

        self.main_tag = main_tag
        self.save_directory = save_directory
        self.data_extractor = data_extractor
        self.writer = SummaryWriter(self.save_directory)

        os.makedirs(save_directory, exist_ok=True)

    def call(self, executor):
        self.__save(executor.trainer.state)

    def __save(self, executor_state):
        self.writer.add_scalars(
            self.main_tag,
            self.data_extractor(executor_state),
            global_step=executor_state.global_step
        )
