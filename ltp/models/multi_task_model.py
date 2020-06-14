#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Tuple, Union, Dict

import torch
from transformers import PretrainedConfig

from .model import Model


class MultiTaskModel(Model, alias='_multi'):
    """
    基本多任务序列标注模型
    """

    def create_decoder(self, input_size, label_num, dropout=0.1, **kwargs):
        """
        封装了各种解码器
        :param decoder: [None=Linear,lan=lan decoder,crf] 默认是简单线性层分类，目前支持 lan decoder

        :param hidden_size: decoder = lan 时必填, lan hidden size
        :param num_heads: decoder = lan 时使用, lan 多头注意力模型 heads，默认为 5
        :param num_layers: decoder = lan 时使用, lan decoder 层数，默认为 3
        :param lan: decoder = lan 时使用，lan decoder 其他参数

        :param arc_hidden_size: decoder = biaffine 时必填
        :param rel_hidden_size: decoder = biaffine 时必填
        :param rel_num: decoder = biaffine 时必填，rel 数目

        :param bias: decoder = Linear 时，传入Linear层
        """
        decoder_type = kwargs.pop('decoder', 'Linear')
        return super(MultiTaskModel, self).create_decoder(
            input_size, label_num, dropout, decoder=decoder_type, **kwargs
        )


class SimpleMultiTaskModel(MultiTaskModel):
    def __init__(self, pretrained: str = None, config: Union[str, PretrainedConfig] = None,
                 dropout=0.1, freeze=False, **kwargs):
        super().__init__()
        self.pretrained = self.create_pretrained(pretrained, config=config, freeze=freeze)
        config = self.pretrained.config
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        self.task = None
        self.decoders_word_base = {}
        self.decoders_use_cls = {}
        for task, decoder_kwargs in kwargs.items():
            self.decoders_word_base[task] = decoder_kwargs.pop('word_base', False)
            self.decoders_use_cls[task] = decoder_kwargs.pop('use_cls', False)
            setattr(self, f"{task}_decoder", self.create_decoder(config.hidden_size, dropout=dropout, **decoder_kwargs))

    def __getitem__(self, item):
        self.task = item
        return self

    def decode(self, pretrained_output, rnn_steps, *args, **kwargs):
        decoder = getattr(self, f"{self.task}_decoder")
        if isinstance(decoder, torch.nn.Linear):
            return decoder(pretrained_output)
        else:
            return decoder(pretrained_output, rnn_steps, *args, **kwargs)

    def forward(self, text: Dict[str, torch.Tensor], *args, **kwargs):
        pretrained_output, *_ = self.pretrained(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            token_type_ids=text['token_type_ids']
        )

        # remove [CLS] [SEP]
        use_cls = self.decoders_use_cls[self.task]
        pretrained_output = torch.narrow(
            pretrained_output, 1,
            1 - use_cls, pretrained_output.size(1) - 2 + use_cls
        )
        pretrained_output = self.emb_dropout(pretrained_output)

        if self.decoders_word_base[self.task]:
            word_idx, word_idx_len = text['word_index'], text['word_length']
            if use_cls:
                cls_tensor = torch.zeros((word_idx.shape[0], 1), dtype=word_idx.dtype, device=word_idx.device)
                word_idx = torch.cat([cls_tensor, word_idx + 1], dim=-1)
            word_idx = word_idx.unsqueeze(-1).expand(-1, -1, pretrained_output.shape[-1])
            pretrained_output = torch.gather(pretrained_output, dim=1, index=word_idx)
            seq_lens = word_idx_len
        else:
            seq_lens = text['text_length']

        return self.decode(pretrained_output, seq_lens, *args, **kwargs)
