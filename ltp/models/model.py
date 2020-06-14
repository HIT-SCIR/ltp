#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Union

from torch.nn import Module, Linear
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
from ltp.core import Registrable
from ltp import modules

"""
说明：
    text: 指代类似生文本，指的的文本，也就是字(Char Based)级别
    word: 指代词语，通常是词语，比如 word_index 就是词语开头的下标
"""


class Model(Module, metaclass=Registrable):
    """
    模型基础类，从本类继承的模型自动注册
    """
    pretrained: PreTrainedModel

    def create_pretrained(self, pretrained: str = None, config: Union[str, dict, PretrainedConfig] = None,
                          freeze: bool = False):
        if isinstance(pretrained, str):  # 认为是 PATH 或 Huggingface 社区模型
            pretrained = AutoModel.from_pretrained(pretrained)
        elif isinstance(config, PretrainedConfig):
            pretrained = AutoModel.from_config(config)
        elif isinstance(config, str) or isinstance(config, dict):  # 认为是 PATH 或 Huggingface 社区模型
            config = AutoConfig.for_model(**config)
            pretrained = AutoModel.from_config(config)
        else:
            raise NotImplementedError()

        if freeze:
            for param in pretrained.parameters():
                param.requires_grad = False
        return pretrained

    def create_decoder(self, input_size, label_num, dropout=0.1, **kwargs):
        """
        封装了各种解码器
        """
        decoder_type = kwargs.pop('decoder', 'Linear')
        if decoder_type:
            decoder_args = kwargs.pop(decoder_type, {})
            decoder_args.setdefault('dropout', dropout)
            decoder_class = modules.Module.by_name(decoder_type)
            return decoder_class(input_size, label_num, **decoder_args)
        else:
            raise NotImplementedError()

    def __getitem__(self, item):
        self.task = item
        return self

    def forward(self, *args, **kwargs):
        """前向传播函数
        """
        raise NotImplementedError()
