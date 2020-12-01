#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import json
from typing import List
from fire import Fire
from ltp import LTP


class Run(object):
    def __init__(self, path: str = 'small', batch_size: int = 50, device: str = None, onnx: bool = False):
        self.ltp = LTP(path=path, device=device, need_config=True)
        self.split = lambda a: map(lambda b: a[b:b + batch_size], range(0, len(a), batch_size))

    def _predict(self, sentences: List[str]):
        result = []
        for sentences_batch in self.split(sentences):
            batch_seg, hidden = self.ltp.seg(sentences_batch)
            batch_pos = self.ltp.pos(hidden)
            batch_ner = self.ltp.ner(hidden)
            batch_srl = self.ltp.srl(hidden)
            batch_dep = self.ltp.dep(hidden)
            batch_sdp = self.ltp.sdp(hidden)

            for sent, seg, pos, ner, srl, dep, sdp in \
                    zip(sentences_batch, batch_seg, batch_pos, batch_ner, batch_srl, batch_dep, batch_sdp):
                result.append({
                    'text': sent,
                    'words': seg, 'pos': pos, 'ner': ner,
                    'srl': srl, 'dep': dep, 'sdp': sdp,
                })

        return result

    def test(self, sentences: List[str] = None):
        if sentences is None:
            sentences = ["我去长江大桥玩。"]
        res = self._predict([sentence.strip() for sentence in sentences])
        print(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False))

    def test_seg(self, sentences: List[str] = None):
        self.ltp.add_words("长江大桥")
        if sentences is None:
            sentences = ["我去长江大桥玩。"]
        seg, hidden = self.ltp.seg(sentences)

        print(seg)

    def test_seged(self, sentences: List[str] = None):
        if sentences is None:
            sentences = ["我去长江大桥玩。"]
        seg, hidden = self.ltp.seg(sentences)
        seged, hidden_seged = self.ltp.seg(seg, is_preseged=True)

        print("SEG: ", seg)
        print("SEGED: ", seged)


if __name__ == '__main__':
    Fire(Run)
