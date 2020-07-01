#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import sys
from time import time
from typing import Iterable, Union, List, Set, Dict

import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm
from ltp.exe.config import Config
from ltp.train.callback import Callback, ValidationCallback
from ltp.train.stop_condition import NoStopping, EarlyStopping
from ltp.eval.metrics import Metric
from ltp.train import Trainer
from ltp.utils import cycle


class Executor(object):
    """
    实际训练器的一个简单包装
    """
    config: Config
    trainer: Trainer
    __callbacks: List
    progressbar_metrics: Set
    stop_condition: Union[EarlyStopping, NoStopping]

    def __init__(self, config: Config):
        self.config = config
        self.stop_condition = EarlyStopping(
            **self.config.early_stopping
        ) if self.config.early_stopping else NoStopping()
        self.__callbacks = []
        self.progressbar_metrics = set()
        self.trainer = Trainer.from_params(self.config.trainer, config=self.config)
        self.tasks = self.config.tasks
        self.epoch_size = self.config.epoch_size

    def train(self, epoch: int = 30):
        epoch = self.config.epoch if epoch is None else epoch

        # =================================== 任务初始化 =================================
        multi_task_dataset = {}
        for name, task in self.tasks.items():
            if not task.config.dataset:
                continue
            if name != 'default':
                task.load()
            train, valid = task.train_dataset(self.config.batch_size)
            valid_callback = ValidationCallback(data_loader=valid, metrics=task.metrics, task=name)
            self.register_callback(valid_callback)
            multi_task_dataset[name] = train

        # =================================== 回调建立 ===================================
        for callback in self.config.callbacks:
            self.register_callback(Callback.from_params(callback))
        # ======================= 开始训练 ===============================================
        self.train_wrapper(multi_task_dataset, epoch, tau=self.config.tau)
        # =================================== 指标评估 ===================================
        self.evaluate()

    def evaluate(self, file: str = None, checkpoint: str = None, task: str = None):
        print("========================= Evaluate ==========================", file=sys.stderr)
        self.tasks['default'].load(self.trainer.state, checkpoint)
        if task is None:
            for name, task_obj in self.tasks.items():
                if not task_obj.config.dataset:
                    continue
                if name != 'default':
                    task_obj.load()
                test = task_obj.load_dataset(file, self.config.batch_size)
                self.evaluate_(test, task_obj.metrics, name)
        else:
            if task != 'default':
                self.tasks[task].load()
            test = self.tasks[task].load_dataset(file, self.config.batch_size)
            self.evaluate_(test, self.tasks[task].metrics, task)

    def predict(self, inputs: str, outputs: str, checkpoint: str = None, task: str = None):
        self.tasks['default'].load(self.trainer.state, checkpoint)
        if task != 'default':
            self.tasks[task].load()
        test = self.tasks[task].load_dataset(inputs, self.config.batch_size)
        self.predict_(test, outputs, task)

    def deploy(self, path: str = 'deploy.model', vocab: str = None):
        deploy_state_dict = {'vocab': vocab}
        for task, task_obj in self.tasks.items():
            task_obj.load()
            if task == 'default':
                deploy_state_dict['model'] = task_obj.model.state_dict()
                deploy_state_dict['model_config'] = task_obj.config.model
                deploy_state_dict['pretrained_config'] = task_obj.model.pretrained.config.to_dict()

            for field_name, field in task_obj.fields:
                if field.is_target and hasattr(field, 'vocab') and hasattr(field.vocab, 'itos'):
                    pad_bias = getattr(field, 'pad_bias', 0)
                    deploy_state_dict[task] = field.vocab.itos[pad_bias:]
        torch.save(deploy_state_dict, path)

    def train_wrapper(self, dataloaders: Dict[str, torch.utils.data.DataLoader], epochs: int = 30, tau: float = 1.0):
        """
        通过给定的 data loader 进行训练，训练会进行到epochs或者 stop condition = True

        :param tau: 放大指数
        :param dataloaders: PyTorch DataLoader
        :param epochs: 训练的最大轮数
        """
        print("========================== Train ============================", file=sys.stderr)
        self.trainer.model.train()  # set the module to training mode
        train_start = time()

        # cycle has a memory leak
        dataiters = {k: cycle(v) for k, v in dataloaders.items()}
        if all(hasattr(v, '__len__') for v in dataloaders.values()):
            dataloader_sizes = {k: len(v) for k, v in dataloaders.items()}
            total_size = sum(v for k, v in dataloader_sizes.items())
            Z = sum(pow(v, tau) for v in dataloader_sizes.values())
            tasknames, sampling_weights = zip(*((k, pow(v, tau) / Z) for k, v in dataloader_sizes.items()))
        else:
            raise NotImplementedError("Dataloader 需要实现 __len__ 方法")

        if self.epoch_size:
            total_size = self.epoch_size

        self.tasks['default'].build_scheduler(epochs * total_size)
        self.tasks['default'].restore(self.trainer.state)

        self.trainer.init(epochs * total_size)
        self.trainer.state.current_epoch += 1

        while self.trainer.state.current_epoch <= epochs and not self.stop_condition(self.trainer.state):
            # ------------------------- EPOCH ----------------------------
            self.trainer.before_train()
            self.train_(total_size, tasknames, sampling_weights, dataiters)
            self.__run_post_epoch_callbacks()
            self.trainer.after_train()
            # ------------------------- EPOCH ----------------------------
            self.trainer.state.current_epoch += 1
        print("train time %.2f" % (time() - train_start))

    def train_(self, total_size, tasknames, sampling_weights, dataiters):
        ofm = 0  # out of memory
        with tqdm(range(total_size), desc=f'Train({self.trainer.state.current_epoch}): ') as epoch_steps:
            for _ in epoch_steps:
                try:
                    taskname = np.random.choice(tasknames, p=sampling_weights)
                    dataiter = dataiters[taskname]
                    batch = next(dataiter)
                    self.trainer.state.last_train_loss = self.trainer.train(batch, task=taskname)
                    self.trainer.state.global_step += 1
                    self.__run_post_iteration_callbacks()

                    postfix = {metric: getattr(self.trainer.state, metric) for metric in
                               self.progressbar_metrics}
                    postfix["loss"] = self.trainer.state.last_train_loss
                    postfix["ofm"] = ofm
                    epoch_steps.set_postfix(postfix)
                except Exception as e:
                    detail = e.args[0]
                    if isinstance(detail, str) and detail.startswith("CUDA out of memory"):
                        ofm += 1
                        epoch_steps.set_postfix({"ofm": ofm})
                        continue
                    raise e

    def evaluate_(self, data_loader: torch.utils.data.DataLoader, metrics: Iterable[Metric], task: str) -> \
            Iterable[Metric]:
        """
        评估一个模型
        :param task: 任务信息
        :param data_loader:  PyTorch DataLoader
        :param metrics:  进行评估的 Metrics
        :return: 计算得到的各项指标
        """
        for metric in metrics:
            metric.clear()

        ofm = 0  # out of memory
        self.trainer.before_eval(task)
        with torch.no_grad(), tqdm(data_loader, desc=f'{task}({self.trainer.state.current_epoch}): ')as pbar:
            for batch in pbar:
                try:
                    x, y_pred, y = self.trainer.eval(batch, task)
                    metric_values = {}
                    for metric in metrics:
                        metric.step(y_pred, y)
                        metric_values.update(metric.compute())
                    if ofm > 0:
                        metric_values['ofm'] = ofm
                    pbar.set_postfix(metric_values)
                except Exception as e:
                    detail = e.args[0]
                    if isinstance(detail, str) and detail.startswith("CUDA out of memory"):
                        ofm += 1
                        continue
                    raise e
        self.trainer.after_eval(task)
        return metrics

    def predict_(self, dataloader: torch.utils.data.DataLoader, outputs: str, task):
        """
        进行预测操作

        :param dataloader: PyTorch DataLoader
        :param outputs: 输出的文件名
        """

        self.trainer.before_predict(task)
        with torch.no_grad(), open(outputs, mode='w', encoding='utf8') as f, \
                tqdm(dataloader, dynamic_ncols=True, desc=f'Predict: ') as pbar:
            for batch in pbar:
                result = self.trainer.predict(batch, task)
                for pred in result:
                    f.writelines("\t".join(pred) + "\n")
        self.trainer.after_predict(task)

    def test(self):
        for name, task in self.tasks.items():
            if not task.config.dataset:
                continue
            if name != 'default':
                task.load()
            train, valid = task.train_dataset(self.config.batch_size, False)

            for name, field in task.fields:
                if hasattr(field, 'vocab') and hasattr(field.vocab, 'itos'):
                    print(name, 'vocab size(with pad):', len(field.vocab.itos), file=sys.stderr)
                    print(field.vocab.itos, file=sys.stderr)

            for train_iter in tqdm(train):
                pass
            for valid_iter in tqdm(valid):
                pass
            for test_iter in tqdm(task.load_dataset(batch_size=self.config.batch_size)):
                pass

    def add_progressbar_metric(self, name):
        self.progressbar_metrics.add(name)

    def register_callback(self, callback: Callback):
        callback.init(self)
        self.__callbacks.append(callback)

    def __run_post_iteration_callbacks(self):
        for callback in self.__callbacks:
            if callback.iteration is None:
                continue
            if callback.iteration != 0 and self.trainer.state.global_step % callback.iteration == 0:
                callback(self)

    def __run_post_epoch_callbacks(self):
        for callback in self.__callbacks:
            if callback.epoch is None:
                continue
            if (callback.epoch != 0 and self.trainer.state.global_step % callback.epoch == 0):
                callback(self)
