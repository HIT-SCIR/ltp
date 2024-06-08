#! /usr/bin/env python
# -*- coding: utf-8 -*
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

"""
LTP Server 是对 LTP 的一个简单包装，依赖于 tornado，使用方式如下：
.. code-block:: bash
    pip install fastapi uvicorn
    uvicorn server:app
"""

from typing import List, Union
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from ltp import LTP


class SRLRole(BaseModel):
    text: str
    offset: int
    length: int
    type: str


class Parent(BaseModel):
    parent: int
    relate: str


class Word(BaseModel):
    id: int
    length: int
    offset: int
    text: str
    pos: str
    parent: int
    relation: str
    roles: List[SRLRole]
    parents: List[Parent]


class NE(BaseModel):
    text: str
    offset: int
    ne: str
    length: int


class Item(BaseModel):
    text: str
    nes: List[NE]
    words: List[Word]


app = FastAPI()

ltp = LTP("LTP/tiny")

if torch.cuda.is_available():
    ltp.to("cuda")


@app.post("/api")
async def predict(sentences: List[str]) -> List[Item]:
    output = ltp.pipeline(sentences, tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])

    # https://github.com/HIT-SCIR/ltp/blob/main/python/interface/docs/quickstart.rst
    # 需要注意的是，在依存句法当中，虚节点ROOT占据了0位置，因此节点的下标从1开始。
    result = []
    for idx, sentence in enumerate(sentences):
        id = 0
        offset = 0
        words = []
        for word, pos, parent, relation in \
                zip(output.cws[idx], output.pos[idx], output.dep[idx]['head'], output.dep[idx]['label']):
            # print([id, word, pos, parent, relation])
            words.append({
                'id': id,
                'length': len(word),
                'offset': offset,
                'text': word,
                'pos': pos,
                'parent': parent - 1,
                'relation': relation,
                'roles': [],
                'parents': []
            })
            id = id + 1
            offset = offset + len(word)

        for token_srl in output.srl[idx]:
            for (argument, text, start, end) in token_srl['arguments']:
                # print(token_srl['index'], token_srl['predicate'], argument)
                offset = words[start]['offset']
                words[token_srl['index']]['roles'].append({
                    'text': text,
                    'offset': offset,
                    'length': len(text),
                    'type': argument
                })

        start = 0
        for end, label in zip(output.sdp[idx]['head'], output.sdp[idx]['label']):
            words[start]['parents'].append({'parent': end - 1, 'relate': label})
            start = start + 1

        nes = []
        for role, text, start, end in output.ner[idx]:
            nes.append({
                'text': text,
                'offset': start,
                'ne': role.lower(),
                'length': len(text)
            })

        result.append(
            {
                'text': sentence,
                'nes': nes,
                'words': words
            }
        )

    return result
