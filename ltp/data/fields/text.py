#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from typing import Union, Dict
from itertools import chain
import torch, torch.nn.utils.rnn as rnn, numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from ltp.core.exceptions import DataUnsupported
from . import Field


class TextField(Field, alias='text'):
    """文本域

    Args:
        tokenizer: Tokenizer
        name: Field Na,e
        return_length: 是否同时返回 length
        word_info: 是否返回 word index
        is_target: 是否为目标域
    """
    tokenizer_cls = AutoTokenizer

    def __init__(self, tokenizer: Union[str, PreTrainedTokenizer, Dict[str, str]],
                 name='text', return_tokens=False, return_word_idn=False, return_length=True,
                 word_info=True, is_target=False):
        super(TextField, self).__init__(name, is_target)
        self.word_info = word_info
        self.return_length = return_length
        self.return_tokens = return_tokens
        self.return_word_idn = return_word_idn

        if isinstance(tokenizer, str):
            self.tokenizer = self.tokenizer_cls.from_pretrained(tokenizer, use_fast=True)
        elif isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
            assert self.tokenizer.is_fast
        elif isinstance(tokenizer, Dict):
            tokenizer['pretrained_model_name_or_path'] = tokenizer.pop('path')
            tokenizer['use_fast'] = True
            self.tokenizer = self.tokenizer_cls.from_pretrained(**tokenizer)

    def preprocess(self, x):
        sentence = self.tokenizer.batch_encode_plus(
            x, add_special_tokens=False, return_attention_masks=False, return_token_type_ids=False
        )
        subword_len = [len(subword.offsets) for subword in sentence.encodings]
        word_lengths = np.cumsum([0] + subword_len, dtype=np.int64)
        word_start, text_length = word_lengths[:-1], word_lengths[-1]
        word_start_idn = list(chain.from_iterable([0] + [1] * (length - 1) for length in subword_len))

        if text_length > 510:
            raise DataUnsupported("文本过长！！")

        # mixed_sentence = ' '.join(x)
        # resplit = self.tokenizer.encode(
        #     mixed_sentence, add_special_tokens=False, return_attention_masks=False, return_token_type_ids=False
        # )
        # if len(resplit) != len(word_start_idn):
        #     print("X: ", x)
        #     print("Mixed: ", mixed_sentence)

        return ' '.join(x), torch.as_tensor(text_length), \
               torch.as_tensor(word_start_idn), torch.as_tensor(word_start), torch.as_tensor(len(x)),

    def process(self, batch, device=None):
        sentence, text_length, word_start_idn, word_index, word_length = zip(*batch)
        tokenized = self.tokenizer.batch_encode_plus(list(sentence), return_tensors='pt')

        res = {
            'input_ids': tokenized['input_ids'].to(device),
            'token_type_ids': tokenized['token_type_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device),
        }
        if self.return_length:
            res['text_length'] = torch.stack(text_length).to(device)
        if self.return_word_idn:
            res['word_idn'] = rnn.pad_sequence(word_start_idn, batch_first=True).to(device)
        if self.word_info:
            res['word_index'] = rnn.pad_sequence(word_index, batch_first=True).to(device)
            if self.return_length:
                res['word_length'] = torch.stack(word_length).to(device)

        return res


class MixedTextField(TextField):
    def process(self, batch, device=None):
        dataset, sentence, text_length, word_start_idn, word_index, word_length = zip(*batch)
        cls = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(dataset), device=device).unsqueeze_(1)
        tokenized = self.tokenizer.batch_encode_plus(list(sentence), return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device=device)[:, 1:]

        res = {
            'input_ids': torch.cat([cls, input_ids], dim=-1),
            'token_type_ids': tokenized['token_type_ids'].to(device),
            'attention_mask': tokenized['attention_mask'].to(device),
        }
        if self.return_length:
            res['text_length'] = torch.stack(text_length).to(device)
        if self.return_word_idn:
            res['word_idn'] = rnn.pad_sequence(word_start_idn, batch_first=True).to(device)
        if self.word_info:
            res['word_index'] = rnn.pad_sequence(word_index, batch_first=True).to(device)
            if self.return_length:
                res['word_length'] = torch.stack(word_length).to(device)

        return res
