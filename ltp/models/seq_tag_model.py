#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Tuple, Union, Dict, Optional

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig

from .model import Model
from ltp.modules import Module


class SequenceTaggingModel(Model, alias="_sequence_tagging"):
    """
    基本序列标注模型，封装了各种解码器，将来可能要进行解耦，如果使用crf解码器，需要换用支持 CRF 的 Trainer

    使用 init_decoder 函数初始化解码器

    :param decoder: [None=Linear,lan=lan decoder] 默认是简单线性层分类，目前支持 lan decoder

    :param hidden_size: decoder = lan 时必填, lan hidden size
    :param num_heads: decoder = lan 时使用, lan 多头注意力模型 heads，默认为 5
    :param num_layers: decoder = lan 时使用, lan decoder 层数，默认为 3
    :param lan: decoder = lan 时使用，lan decoder 其他参数

    :param arc_hidden_size: decoder = biaffine 时必填
    :param rel_hidden_size: decoder = biaffine 时必填
    :param rel_num: decoder = biaffine 时默认为label_num，rel 数目

    :param bias: decoder = Linear 时，传入Linear层
    """

    decoder: Module

    def init_decoder(self, input_size, label_num, dropout=0.1, **kwargs):
        """
        基本序列标注模型，封装了各种解码器
        """
        self.decoder = self.create_decoder(input_size, label_num, dropout, **kwargs)

    def hidden2logits(self, logits, seq_len, gold: Optional = None):
        return self.decoder.forward(logits, seq_len, gold)

    def forward_(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("未实现")

    def forward(self, text: Dict[str, torch.Tensor], gold: Optional = None):
        features, seq_lens = self.forward_(text)
        return self.hidden2logits(features, seq_lens, gold)


class SimpleTaggingModel(SequenceTaggingModel, alias="SimpleTagging"):
    """
    基本序列标注模型

    :param pretrained: 预训练模型路径或名称，参照 huggingface/transformers
    :param config: 预训练模型路径或名称，参照 huggingface/transformers
    :param label_num: 分类标签数目
    :param dropout: pretrained Embedding dropout，默认0.1
    :param word_base: 是否是以词为基础的模型，如果是，输入时需要传入 word index
    :param decoder: [None=Linear,lan=lan decoder] 默认是简单线性层分类，目前支持 lan decoder

    :param hidden_size: decoder = lan 时必填, lan hidden size
    :param num_heads: decoder = lan 时使用, lan 多头注意力模型 heads，默认为 5
    :param num_layers: decoder = lan 时使用, lan decoder 层数,，默认为 3
    :param lan: decoder = lan 时使用，lan decoder 其他参数

    :param arc_hidden_size: decoder = graph 时必填
    :param rel_hidden_size: decoder = graph 时必填
    :param rel_num: decoder = graph 时默认为label_num，rel 数目

    :param bias: decoder = Linear 时，传入Linear层
    """

    def __init__(self, label_num, pretrained: str = None, config: Union[str, PretrainedConfig] = None,
                 dropout=0.1, word_base=False, use_cls=False, freeze=False, **kwargs):
        """
        基本序列标注模型
        """
        super(SimpleTaggingModel, self).__init__()
        self.word_base = word_base
        self.use_cls = use_cls
        self.gard_enable = not freeze
        self.pretrained = self.create_pretrained(pretrained, config=config, freeze=freeze)
        config = self.pretrained.config
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        self.init_decoder(
            input_size=config.hidden_size,
            label_num=label_num,
            dropout=dropout,
            **kwargs
        )

    def forward_(self, text: Dict[str, torch.Tensor]):
        with torch.set_grad_enabled(self.gard_enable):
            pretrained_output, *_ = self.pretrained(
                text['input_ids'],
                attention_mask=text['attention_mask'],
                token_type_ids=text['token_type_ids']
            )
            # remove [CLS] [SEP]
            pretrained_output = torch.narrow(
                pretrained_output, 1,
                1 - self.use_cls, pretrained_output.size(1) - 2 + self.use_cls
            )
            pretrained_output = self.emb_dropout(pretrained_output)
            if self.word_base:
                word_idx = text['word_index']
                if self.use_cls:
                    cls_tensor = torch.zeros((word_idx.shape[0], 1), dtype=word_idx.dtype, device=word_idx.device)
                    word_idx = torch.cat([cls_tensor, word_idx + 1], dim=-1)
                word_idx = word_idx.unsqueeze(-1).expand(-1, -1, pretrained_output.shape[-1])
                pretrained_output = torch.gather(pretrained_output, dim=1, index=word_idx)
                seq_lens = text['word_length']
            else:
                seq_lens = text['text_length']
            return pretrained_output, seq_lens
