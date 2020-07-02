#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import List

import os
import torch
import numpy as np
from ltp.utils import length_to_mask, is_chinese_char
from ltp.utils.seqeval import get_entities
from ltp import LTP
from ltp.ltp import WORD_MIDDLE, no_gard


def convert(item: list):
    array = np.asarray(item, dtype=np.int64)

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
    return array


class FastLTP(LTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import onnxruntime as rt
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # fixme should auto detect
        providers = ['CPUExecutionProvider'] if self.device.type == 'cpu' else ['GPUExecutionProvider']

        onnx_path = os.path.join(self.cache_dir, "ltp.onnx")
        if not os.path.isfile(onnx_path):
            self.pretrained_export(onnx_path)

        self.onnx = rt.InferenceSession(onnx_path, so, providers=providers)

    def pretrained_export(self, path: str):
        from torch.onnx import export

        dummy_input = {
            'input_ids': torch.as_tensor([
                [101, 800, 1373, 3739, 1990, 1343, 2897, 1912, 6132, 511, 102, 0, 0],
                [101, 2571, 3647, 4638, 1383, 2094, 1355, 7942, 5445, 1318, 3289, 511, 102]
            ], device=self.device),
            'token_type_ids': torch.as_tensor([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ], device=self.device),
            'attention_mask': torch.as_tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 11
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 13
            ], device=self.device),
        }

        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        output_names = ['hidden', 'seg']
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'length'},
            'attention_mask': {0: 'batch', 1: 'length'},
            'token_type_ids': {0: 'batch', 1: 'length'},
            'hidden': {0: 'batch', 1: 'length'},
            'seg': {0: 'batch', 1: 'length'},
        }
        model_args = tuple(dummy_input[arg] for arg in input_names)

        class Model(torch.nn.Module):
            def __init__(self, pretrained, seg):
                super().__init__()
                self.pretrained = pretrained
                self.seg = seg

            def forward(self, *args, **kwargs):
                hidden = self.pretrained(*args, **kwargs)[0]

                cls = hidden[:, :1]
                hidden_cut = hidden[:, 1:-1]
                seg = self.seg(hidden_cut)
                seg = torch.argmax(seg, dim=-1)

                return cls, hidden_cut, seg

        model = Model(
            self.model.pretrained,
            self.model.seg_decoder
        )

        with torch.no_grad():
            export(
                model,
                model_args,
                f=path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                use_external_data_format=False,
                enable_onnx_checker=True,
                opset_version=12,
            )

    @no_gard
    def seg(self, inputs: List[str]):
        tokenizerd = self.tokenizer.batch_encode_plus(inputs, padding=True)
        pretrained_inputs = {key: convert(value) for key, value in tokenizerd.items()}
        length = np.sum(pretrained_inputs['attention_mask'], axis=-1) - 2

        # todo: io binding
        cls, hidden, seg = self.onnx.run(None, pretrained_inputs)

        segment_output = self._convert_idx_to_name(seg, length, self.seg_vocab)

        word_cls = torch.as_tensor(cls, device=self.device)
        char_input = torch.as_tensor(hidden, device=self.device)

        # todo: performance
        sentences = []
        word_idx = []
        word_length = []
        for source_text, encoding, sentence_seg_tag in zip(inputs, tokenizerd.encodings, segment_output):
            text = [source_text[start:end] for start, end in encoding.offsets[1:-1] if end != 0]

            last_word = 0
            for idx, word in enumerate(encoding.words[1:-1]):
                if word is None or is_chinese_char(text[idx][-1]):
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
