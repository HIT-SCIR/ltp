#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import json
from typing import List

from fire import Fire

from ltp import LTP, FastLTP


class Run(object):
    def __init__(self, path: str = 'small', batch_size: int = 50, device: str = None, onnx: bool = False):
        if onnx:
            self.ltp = FastLTP(path=path, device=device, need_config=True)
        else:
            self.ltp = LTP(path=path, device=device, need_config=True)
        self.split = lambda a: map(lambda b: a[b:b + batch_size], range(0, len(a), batch_size))

    def _build_words(self, words, pos, dep):
        res = [{'id': -1, 'length': 0, 'offset': 0, 'text': 'root'}]
        for word, p, (id, parent, relation) in zip(words, pos, dep):
            offset = res[-1]['offset'] + res[-1]['length']
            res.append({
                'id': id - 1,
                'length': len(word),
                'offset': offset,
                'text': word,
                'pos': p,
                'parent': parent - 1,
                'relation': relation,
                'roles': [],
                'parents': []
            })

        return res[1:]

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

                words = self._build_words(seg, pos, dep)

                for word, token_srl in zip(words, srl):
                    for role, start, end in token_srl:
                        text = "".join(seg[start:end + 1])
                        offset = words[start]['offset']
                        word['roles'].append({
                            'text': text,
                            'offset': offset,
                            'length': len(text),
                            'type': role
                        })

                for start, end, label in sdp:
                    words[start - 1]['parents'].append({'parent': end - 1, 'relate': label})

                nes = []
                for role, start, end in ner:
                    text = "".join(seg[start:end + 1])
                    nes.append({
                        'text': text,
                        'offset': start,
                        'ne': role.lower(),
                        'length': len(text)
                    })

                result.append({
                    'text': sent,
                    'nes': nes,
                    'words': words
                })

        return result

    def test(self, sentences: List[str] = None):
        self.ltp.add_words("DMI与主机通讯中断")
        if sentences is None:
            sentences = [
                "我们都是中国人。",
                "遇到苦难不要放弃，加油吧！奥利给！",
                "乔丹是一位出生在纽约的美国职业篮球运动员。"
            ]
        res = self._predict([sentence.strip() for sentence in sentences])
        print(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False))

    def save(self, out='ltp.npz'):
        import numpy as np
        nps = {}
        for k, v in self.ltp.model.state_dict().items():
            k = k.replace("gamma", "weight").replace("beta", "bias")
            nps[k] = np.ascontiguousarray(v.cpu().numpy())

        np.savez(out, **nps)

        config = self.ltp.config
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def test_seged(self):
        import torch
        sentences = [
            'My name is tom.',
            'He called Tom to get coats.',
            '他叫Tom去拿外衣。',
            '他叫汤姆去拿外衣。',
            "我去长江大桥玩。"
        ]
        seg, hidden = self.ltp.seg(sentences)
        seged, hidden_seged = self.ltp.seg(seg, is_preseged=True)
        hidden: dict
        hidden_seged: dict
        for key, value in hidden.items():
            if isinstance(value, torch.Tensor):
                test = torch.sum(value.float() - hidden_seged[key].float()).numpy()
                print(key, test)

        print(seg == seged)


if __name__ == '__main__':
    Fire(Run)
