#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import List

import torch
from ltp.utils import length_to_mask
from ltp.utils.seqeval import get_entities
import numpy as np
from ltp import LTP
from ltp.ltp import WORD_MIDDLE, no_gard
from ltp.utils.sent_split import split_sentence
import itertools


def convert(item: list):
    array = np.asarray(item, dtype=np.int64)

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
    return array


class FastLTP(LTP):
    def __init__(self, *args, onnx: str, **kwargs):
        super().__init__(*args, **kwargs)
        import onnxruntime as rt
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = ['CPUExecutionProvider'] if self.device.type == 'cpu' else ['GPUExecutionProvider']
        self.onnx = rt.InferenceSession(onnx, so, providers=providers)

    @no_gard
    def seg(self, inputs: List[str]):
        length = [len(text) for text in inputs]
        tokenizerd = self.tokenizer.batch_encode_plus(inputs, pad_to_max_length=True)
        pretrained_inputs = {key: convert(value) for key, value in tokenizerd.items()}
        cls, hidden, seg = self.onnx.run(None, pretrained_inputs)

        segment_output = self._convert_idx_to_name(seg, length, self.seg_vocab)

        word_cls = torch.as_tensor(cls, device=self.device)
        char_input = torch.as_tensor(hidden, device=self.device)

        sentences = []
        word_idx = []
        word_length = []
        for source_text, encoding, sentence_seg_tag in zip(inputs, tokenizerd.encodings, segment_output):
            text = [source_text[start:end] for start, end in encoding.offsets[1:-1] if end != 0]

            last_word = 0
            for idx, word in enumerate(encoding.words[1:-1]):
                if word is None or self._is_chinese_char(text[idx][-1]):
                    continue
                if word != last_word:
                    text[idx] = ' ' + text[idx]
                    last_word = word
                else:
                    sentence_seg_tag[idx] = WORD_MIDDLE

            entities = get_entities(sentence_seg_tag)
            word_length.append(len(entities))

            sentences.append([''.join(text[entity[1]:entity[2] + 1]).lstrip() for entity in entities])
            word_idx.append(torch.as_tensor([entity[1] for entity in entities], device=self.device))

        word_idx = torch.nn.utils.rnn.pad_sequence(word_idx, batch_first=True)
        word_idx = word_idx.unsqueeze(-1).expand(-1, -1, char_input.shape[-1])
        word_input = torch.gather(char_input, dim=1, index=word_idx)

        word_cls_input = torch.cat([word_cls, word_input], dim=1)
        word_cls_mask = length_to_mask(torch.as_tensor(word_length, device=self.device) + 1)
        word_cls_mask[:, 0] = False  # ignore the first token of each sentence

        return sentences, {
            'word_cls': word_cls, 'word_input': word_input, 'word_length': word_length,
            'word_cls_input': word_cls_input, 'word_cls_mask': word_cls_mask
        }
