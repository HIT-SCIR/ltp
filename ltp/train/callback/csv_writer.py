#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os
import csv
import time

from . import Callback

default_save_directory = './log'
default_filename = 'log.csv'


class CsvWriter(Callback, alias="csv"):
    def __init__(self, iteration, epoch, save_directory=default_save_directory, filename=default_filename,
                 delimiter=',', extra_header=None, data_extractor=lambda x: list()):
        super(CsvWriter, self).__init__(iteration, epoch)
        file, ext = filename.rsplit('.', 1)
        self.log_file_path = os.path.join(save_directory, file + '_' + time.strftime("%Y%m%d_%H%M%S") + '.' + ext)
        self.delimiter = delimiter
        self.data_extractor = data_extractor

        os.makedirs(save_directory, exist_ok=True)

        with open(self.log_file_path, mode='w') as writer:
            self.header = ['timestamp', 'epoch', 'iteration', 'train loss']
            if extra_header is not None:
                if not isinstance(extra_header, list):
                    raise TypeError("extra_header should be a list.")
                self.header += extra_header

            writer = csv.writer(writer, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.header)

    def call(self, executor):
        self.__save(executor.trainer.state)

    def __save(self, executor_state):
        with open(self.log_file_path, mode='a') as writer:
            writer = csv.writer(writer, delimiter=self.delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            extra_data = self.data_extractor(executor_state)
            data = [time.time(),
                    executor_state.current_epoch + 1,
                    executor_state.current_iteration + 1,
                    executor_state.last_train_loss] + extra_data
            assert len(data) == len(self.header)
            writer.writerow(data)
