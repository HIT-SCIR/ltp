#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import List

import torch
from torch import Tensor
from .eisner import eisner
from .initial import initial_parameter
from .clip_grad_norm import clip_grad_norm
from .deprecated import deprecated, deprecated_param
from .sent_split import split_sentence
from .ltp_trie import Trie


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def length_to_mask(length: Tensor, max_len: int = None, dtype=None):
    """
    将 Sequence length 转换成 Mask

    >>> lens = [3, 5, 4]
    >>> length_to_mask(length)
    >>> [[1, 1, 1, 0, 0],\
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 0]]

    :param length: [batch,]
    :param max_len: 最大长度
    :param dtype: nn.dtype
    :return: batch * max_len : 如果 max_len is None
    :return: batch * max(length) : 如果 max_len is None
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def select_logits_with_mask(logits, mask):
    if len(logits.shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(logits).to(torch.bool)
        logits_select = torch.masked_select(logits, mask).view(-1, logits.size(-1))
    else:
        logits_select = logits  # Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
    return logits_select


def expand_bio(word_lens: List[int], srl: List[str]):
    res = []
    for word_len, tag in zip(word_lens, srl):
        if tag.startswith('B-'):
            res.extend([tag] + ['I-' + tag[2:]] * (word_len - 1))
        else:
            res.extend([tag] * word_len)
    return res


def pad_sequence(sequences, batch_first=True, pad_value=0):
    def length(sequence):
        if isinstance(sequence, torch.Tensor):
            return sequence.size(0)
        if isinstance(sequence, list):
            return len(sequence)

    lengths, sequences = zip(*[(length(sequence), torch.as_tensor(sequence)) for sequence in sequences])

    return torch.as_tensor(lengths), torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=batch_first, padding_value=pad_value)


try:
    from ltp_plugin import get_entities, is_chinese_char, segment_decode

    USE_PLUGIN = False

except Exception as e:
    from .seqeval import get_entities

    USE_PLUGIN = False


    def segment_decode(inputs, segment_output, offsets, words):
        pass


    def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(cp)
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False
