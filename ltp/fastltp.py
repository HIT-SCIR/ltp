#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os
import torch
import numpy as np
from ltp import LTP
from ltp.ltp import no_gard, TensorType


def convert(item: list):
    array = np.asarray(item, dtype=np.int64)

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
    return array


class FastLTP(LTP):
    tensor = TensorType.NUMPY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import onnxruntime as rt
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # fixme should auto detect
        providers = ['CPUExecutionProvider'] if self.device.type == 'cpu' else ['GPUExecutionProvider']

        onnx_path = os.path.join(self.cache_dir, "ltp.onnx")
        if not os.path.isfile(onnx_path):
            ltp_onnx_export(self, onnx_path)

        self.onnx = rt.InferenceSession(onnx_path, so, providers=providers)

    def __str__(self):
        return f"FastLTP {self.version} on {self.device}"

    def __repr__(self):
        return f"FastLTP {self.version} on {self.device}"

    @no_gard
    def _seg(self, tokenizerd):
        pretrained_inputs = {key: convert(value) for key, value in tokenizerd.items()}
        length = np.sum(pretrained_inputs['attention_mask'], axis=-1) - 2

        # todo: io binding
        cls, hidden, seg = self.onnx.run(None, pretrained_inputs)

        word_cls = torch.as_tensor(cls, device=self.device)
        char_input = torch.as_tensor(hidden, device=self.device)
        return word_cls, char_input, seg, length


def ltp_onnx_export(ltp: LTP, path: str):
    from torch.onnx import export

    dummy_input = {
        'input_ids': torch.as_tensor([
            [101, 800, 1373, 3739, 1990, 1343, 2897, 1912, 6132, 511, 102, 0, 0],
            [101, 2571, 3647, 4638, 1383, 2094, 1355, 7942, 5445, 1318, 3289, 511, 102]
        ], device=ltp.device),
        'token_type_ids': torch.as_tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], device=ltp.device),
        'attention_mask': torch.as_tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 11
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 13
        ], device=ltp.device),
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
        ltp.model.pretrained,
        ltp.model.seg_decoder
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
