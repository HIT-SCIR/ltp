#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Dict
import torch
import toml

from ltp.exe.task import TaskConfig, Task


class Config(object):
    """
    全局配置文件
    """
    tasks: Dict[str, Task]
    temperature_scheduler: dict
    tau: float

    def __init__(self, config_path, device=None):
        config = toml.load(config_path)
        if device is None:
            self.device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        global_config = config.pop("Global", {})
        self.verbose = global_config.pop("verbose", False)
        self.tau = global_config.pop("tau", 1.0)
        self.batch_size = global_config.pop("batch_size", 5)
        self.torch_seed = global_config.pop('seed', None)
        self.epoch = global_config.pop("epoch", 10)
        self.early_stopping = global_config.pop('early_stopping', None)
        self.epoch_size = global_config.pop('epoch_size', None)

        self.grad_norm = global_config.pop("grad_norm", None)
        self.norm_type = global_config.pop("norm_type", None)
        self.pretrained_grad_norm = global_config.pop("pretrained_grad_norm", None)

        self.callbacks = config.pop("Callbacks", [])

        self.tasks = {}
        # task special
        tasks = config.pop("Task", None)
        self.tasks['default'] = Task(TaskConfig(config, self), self.device)
        if tasks:
            for task in tasks:
                self.tasks[task['name']] = Task(TaskConfig(toml.load(task['config']), self), self.device)

        self.trainer = config.pop(
            'Trainer', {'class': 'default'} if len(self.tasks) == 1 else {'class': 'multi'}
        )

        # 其他的配置都是任务特有的
        for key, value in config.items():
            setattr(self, self._camel_to_underline(key), value)

    def _camel_to_underline(self, camel_format):
        """
        驼峰命名格式转下划线命名格式
        """
        lst = []
        for index, char in enumerate(camel_format):
            if char.isupper() and index != 0:
                lst.append("_")
            lst.append(char)

        return "".join(lst).lower()
