#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import List, Optional

import conllu
from fire import Fire
from itertools import zip_longest

from ltp import LTP, FastLTP
from ltp.data.utils import iter_lines


class Conllu(object):
    """
    :param path: 模型路径，或者自动从网上下载 ['base', 'small', 'tiny']
    :param batch_size: 最大 Batch Size 自动切分
    :param device: ['cpu', 'cuda']
    :param onnx: 是否启用 onnx
    """

    def __init__(self, path: str = 'small', batch_size: int = 50, device: str = None, onnx: bool = False):

        if onnx:
            self.ltp = FastLTP(path=path, device=device, need_config=True)
        else:
            self.ltp = LTP(path=path, device=device, need_config=True)
        self._split = lambda a: map(lambda b: a[b:b + batch_size], range(0, len(a), batch_size))

    def _predict(self, sentences: List[str], pos=True, ner=True, srl=True, dep=True, sdp=True):
        result = []
        for sentences_batch in self._split(sentences):
            batch_seg, hidden = self.ltp.seg(sentences_batch)

            batch_size = len(sentences_batch)
            batch_pos = self.ltp.pos(hidden) if pos else ([[]] * batch_size)
            batch_ner = self.ltp.ner(hidden) if ner else ([None] * batch_size)
            batch_srl = self.ltp.srl(hidden, keep_empty=False) if srl else ([None] * batch_size)
            batch_dep = self.ltp.dep(hidden) if dep else ([None] * batch_size)
            batch_sdp = self.ltp.sdp(hidden) if sdp else ([None] * batch_size)

            result += list(zip(batch_seg, batch_pos, batch_ner, batch_dep, batch_sdp, batch_srl))

        return result

    def predict(
            self, input: str, output: Optional[str] = None,
            pos: bool = True, ner: bool = False, srl: bool = False, dep: bool = True, sdp: bool = False
    ):
        """
        预测文本并输出为 conllu 格式
        :param input: 要预测的文件，每行一句话
        :param output: 输出的结果文件，默认是输入文件添加 .conll 后缀
        :param pos: 是否输出 词性标注 结果 ['True','False']
        :param ner: 是否输出 命名实体识别 结果 ['True','False'], 占用 conllu feats 列
        :param srl: 是否输出 语义角色标注 结果 ['True','False'], 占用 conllu misc 列
        :param dep: 是否输出 依存句法分析 结果 ['True','False']
        :param sdp: 是否输出 语义依存分析 结果 ['True','False']
        """
        if output is None:
            output = f"{input}.conllu"

        with open(output, mode='w', encoding='utf-8') as f:
            sentences = sum([sent for idx, sent in iter_lines(input)], [])
            results = self._predict(sentences, pos, ner, srl, dep, sdp)

            for text, (seg_s, pos_s, ner_s, dep_s, sdp_s, srl_s) in zip(sentences, results):
                tokens = conllu.TokenList([
                    conllu.models.Token(
                        id=idx + 1,
                        form=token,
                        lemma=token,
                        upos=pos if pos else '_',
                        xpos=pos if pos else '_',
                        feats='O' if ner else '_',
                        head=idx,
                        deprel='_',
                        deps='' if sdp else '_',
                        misc='SpaceAfter=No'
                    )
                    for idx, (token, pos) in enumerate(zip_longest(seg_s, pos_s))
                ], conllu.models.Metadata(text=text))

                if ner:
                    for tag, start, end in ner_s:
                        tokens[start]['feats'] = f'B-{tag}'
                        for i in range(start + 1, end):
                            tokens[start]['feats'] = f'I-{tag}'
                if dep:
                    for id, head, tag in dep_s:
                        tokens[id - 1]['head'] = head
                        tokens[id - 1]['deprel'] = tag
                if sdp:
                    for id, head, tag in sdp_s:
                        if tokens[id - 1]['deps']:
                            tokens[id - 1]['deps'] = tokens[id - 1]['deps'] + f"|{head}:{tag}"
                        else:
                            tokens[id - 1]['deps'] = f"{head}:{tag}"

                if srl:
                    srl_predicate, srl_roles = list(zip(*srl_s))
                    srl_predicate_num = len(srl_predicate)
                    if srl_predicate_num > 0:
                        srl_misc = [[f'Predicate={"Y" if i in srl_predicate else "_"}', ['O'] * srl_predicate_num] for i
                                    in
                                    range(len(tokens))]
                        for idx, srl_role in enumerate(srl_roles):
                            for tag, start, end in srl_role:
                                srl_misc[start][-1][idx] = f'B-{tag}'
                                for i in range(start + 1, end):
                                    srl_misc[start][-1][idx] = f'I-{tag}'
                        srl_misc = ["|".join([s[0], "Role=" + ",".join(s[-1])]) for s in srl_misc]

                        for token, misc in zip(tokens, srl_misc):
                            token['misc'] = f"{token['misc']}|{misc}"

                f.write(tokens.serialize())


if __name__ == '__main__':
    Fire(Conllu)
