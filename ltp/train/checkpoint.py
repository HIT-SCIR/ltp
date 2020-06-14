#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import sys
import os, regex, logging
from typing import List, Dict
from collections import OrderedDict

import torch

from ltp.core import Registrable
from ltp.const import checkpoint

default_save_diretory = './checkpoint'
default_best_filename = 'best.pt.tar'

logger = logging.getLogger(__name__)


class CheckpointPlugin(metaclass=Registrable):
    """
    Checkpoint

    :param directory: 存放目录
    :param filename: 检查点文件名
    """

    def __init__(self, directory: str = None, filename: str = None):

        if directory is None:
            directory = default_save_diretory
        if filename is None:
            filename = checkpoint

        self.directory = directory
        self.filename = filename
        os.makedirs(self.directory, exist_ok=True)

    def save(self, task, state):
        raise NotImplementedError

    def load(self, task, state):
        raise NotImplementedError

    def save_checkpoint(self, task, state, filename: str):
        filepath = os.path.join(self.directory, filename)
        ckpt_dict = {
            'model_state_dict': task.model.state_dict(),
            'optimizer_state_dict': task.optimizer.state_dict(),
            'executor_state': state,
            'check_model_class': str(task.model.__class__),
            'check_optimizer_class': str(task.optimizer.__class__),
            # 'check_ldtp_version': __version__    # TODO
        }

        if task.scheduler:
            ckpt_dict['scheduler_state_dict'] = task.scheduler.state_dict()
            ckpt_dict['check_scheduler_class'] = str(task.scheduler.__class__)

        for name, field in task.fields:
            if not hasattr(field, 'use_vocab') or not field.use_vocab:
                continue
            if hasattr(field, 'vocab') and hasattr(field.vocab, '__getstate__'):
                ckpt_dict[name] = field.vocab.__getstate__()

        torch.save(ckpt_dict, filepath)
        logger.debug(f"Have saved checkpoint to {filepath}!!!")
        return filepath

    def load_checkpoint(self, task, state, filename):
        filepath = os.path.join(self.directory, filename)

        model = task.model
        optimizer = task.optimizer

        if not os.path.isfile(filepath):
            logger.warning(f"Restored From checkpoint {filepath} failed!!!")
            return

        checkpoint = torch.load(filepath, map_location=task.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if state:
            state.load_state_dict(checkpoint['executor_state'])

        if task.scheduler:
            task.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        for name, field in task.fields:
            if not hasattr(field, 'use_vocab') or not field.use_vocab:
                continue
            field.build_vocab()
            if hasattr(field, 'vocab') and hasattr(field.vocab, '__setstate__'):
                field.vocab.__setstate__(checkpoint[name])

        logger.info(f"Have restored from checkpoint({filepath})")


class CheckpointManager(object):
    plugins: Dict[str, CheckpointPlugin]

    def __init__(self, directory=None, filename=None, plugins: List = None, **kwargs):
        """
        Checkpoint 管理器

        :param directory: Ckpt 保存路径
        :param filename: 保存的文件名，通常会被转成不同的形式
        :param plugins: List [best, restore]
        :param best: dict 传给 best 的额外参数，使用 best 插件时，metric 必选
        """
        if directory is None:
            directory = default_save_diretory
        self.directory = directory
        self.filename = filename
        self.plugins = OrderedDict()

        if plugins is not None:
            for plugin_name in plugins:
                if not hasattr(self, 'mode'):
                    self.mode = plugin_name
                plugin_class = CheckpointPlugin.by_name(plugin_name)
                plugin_config = kwargs.get(plugin_name, {})
                plugin_config.setdefault('filename', filename)
                plugin_config.setdefault('directory', directory)
                self.plugins[plugin_name] = plugin_class(**plugin_config)
        elif kwargs is None or not len(kwargs):
            self.mode = 'restore'
            self.plugins['restore'] = RestoreCheckpointPlugin(directory=directory, filename=filename)
        else:
            for plugin_name, plugin_config in kwargs.items():
                if not hasattr(self, 'mode'):
                    self.mode = plugin_name
                plugin_class = CheckpointPlugin.by_name(plugin_name)
                plugin_config.setdefault('filename', filename)
                plugin_config.setdefault('directory', directory)
                self.plugins[plugin_name] = plugin_class(**plugin_config)

    def __getitem__(self, item):
        if item in self.plugins:
            self.mode = item
        else:
            self.mode = LoadCheckpointPlugin(directory=self.directory, filename=item)
        return self

    def load(self, task, state):
        # 加载模型
        if isinstance(self.mode, str):
            self.plugins[self.mode].load(task, state)
        elif isinstance(self.mode, LoadCheckpointPlugin):
            self.mode.load(task, state)

    def save(self, task, state):
        for plugin in self.plugins.values():
            plugin.save(task, state)


class LoadCheckpointPlugin(CheckpointPlugin, alias='_load'):
    def load(self, task, state):
        self.load_checkpoint(task, state, self.filename)

    def save(self, task, state):
        pass


class BestCheckpointPlugin(CheckpointPlugin, alias='best'):
    def __init__(self, metric, directory: str = None, filename: str = None):
        if filename is None:
            filename = default_best_filename
        super(BestCheckpointPlugin, self).__init__(directory=directory, filename=filename)
        self.metric_name = metric
        self.best_metric = None

        c = self.filename.count('.')
        base, *ext = self.filename.rsplit('.', c)
        self.format = f"{base}_%.4f." + '.'.join(ext)
        # base_{metric}.ext
        self.regex = regex.compile(f"{base}_(\\d+\\.\\d+)\\." + '\\.'.join(ext))
        self.last_filepath = None

    def load(self, task, state):
        if not os.path.isdir(self.directory):
            return
        filenames = os.listdir(self.directory)

        best_metric = None
        best_checkpoint = None
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            match = regex.fullmatch(self.regex, filename)
            if match is None:
                continue
            found_metric = float(match.group(1))

            if best_metric is None or found_metric > best_metric:
                best_metric = found_metric
                best_checkpoint = filename

        if best_checkpoint is not None:
            self.load_checkpoint(task, state, best_checkpoint)

    def save(self, task, state):
        try:
            if isinstance(self.metric_name, list):
                metric = sum([state.get(metric_name) for metric_name in self.metric_name]) / len(self.metric_name)
            else:
                metric = state.get(self.metric_name)
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric = metric
                filename = self.format % self.best_metric
                if self.last_filepath is not None and os.path.exists(self.last_filepath):
                    os.remove(self.last_filepath)
                self.last_filepath = self.save_checkpoint(task, state, filename)
        except Exception as e:
            print(f"No metric {self.metric_name}")


class RestoreCheckpointPlugin(CheckpointPlugin, alias="restore"):
    def __init__(self, *args, **kwargs):
        super(RestoreCheckpointPlugin, self).__init__(*args, **kwargs)
        c = self.filename.count('.')
        base, *ext = self.filename.rsplit('.', c)
        self.format = f"{base}_%d." + '.'.join(ext)
        self.regex = regex.compile(f"{base}_(\\d+)\\." + '\\.'.join(ext))

    def load(self, task, state):
        last_iter = None
        last_checkpoint = None
        if not os.path.isdir(self.directory):
            return
        filenames = os.listdir(self.directory)
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            match = regex.fullmatch(self.regex, filename)
            if match is None:
                continue

            found_epoch = int(match.group(1))
            if last_iter is None or found_epoch > last_iter:
                last_iter = found_epoch
                last_checkpoint = filename

        if last_checkpoint is not None:
            self.load_checkpoint(task, state, last_checkpoint)

    def save(self, task, state):
        filename = self.format % state.global_step
        self.save_checkpoint(task, state, filename)
