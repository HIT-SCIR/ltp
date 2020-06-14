#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import signal
import fire
from ltp.exe import Config, Executor


class Command(object):
    """
    命令行交互指令
    """
    config: Config
    executor: Executor

    def __init__(self, config, device=None):
        self.config = Config(config, device)
        if self.config.torch_seed is not None:
            self.setup_seed(self.config.torch_seed)
        self.executor = Executor(config=self.config)
        signal.signal(signal.SIGINT, self.__graceful_exit)

    def setup_seed(self, seed: int):
        import torch
        import numpy as np
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def train(self, epoch: int = None):
        """
        进行训练操作

        :param epoch: 训练的轮数，默认10轮或者从配置文件中指定
        """
        self.executor.train(epoch)

    def eval(self, file: str = None, checkpoint: str = None, task: str = None):
        """
        进行 Evaluation 操作

        :param task: 任务，默认为default
        :param checkpoint: 使用的 checkpoint，默认是采用Best ckpt（需要启用Checkpoint Mannger best plugin）
        :param file: 要进行 evaluation 的文件，默认是test数据集
        """
        self.executor.evaluate(file, checkpoint, task=task)

    def predict(self, input: str, output: str, checkpoint: str = None, task: str = 'default'):
        """
        进行预测操作

        :param task: 任务，默认为default
        :param input: 输入文件
        :param output: 输出文件
        :param checkpoint:  使用的 checkpoint，默认是采用Best ckpt（需要启用Checkpoint Mannger best plugin）
        """
        self.executor.predict(input, output, checkpoint, task=task)

    def deploy(self, path: str = 'deploy.model', vocab: str = None):
        """
        将模型中不需要的部分都清理掉，仅仅保留需要预测的部分
        :param path: 最终保存的模型的路径
        :param vocab: 词典名字，从 huggingface 加载
        """
        self.executor.deploy(path, vocab)

    def test(self):
        """
        测试配置文件是否正确，目前只对Dataset进行验证。
        """
        self.executor.test()

    def __graceful_exit(self, signum, frame):
        print("Sig %s caught. Graceful exit has been called. Currently running epoch will be finished." % signum)
        self.executor.stop_condition = lambda state: True


if __name__ == '__main__':
    fire.Fire(Command)
