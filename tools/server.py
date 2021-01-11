#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
"""
LTP Server 是对 LTP 的一个简单包装，依赖于 tornado，使用方式如下：

.. code-block:: bash

    pip install ltp, tornado
    python utils/server.py serve
"""
import sys
import json
import logging
from typing import List

from tornado import ioloop
from tornado.httpserver import HTTPServer
from tornado.web import Application, RequestHandler
from tornado.log import app_log, gen_log, access_log, LogFormatter
from fire import Fire

from ltp import LTP


class LTPHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.set_header('Content-Type', 'application/json;charset=UTF-8')

    def initialize(self, ltp):
        self.set_default_headers()
        self.ltp = ltp

    def post(self):
        try:
            text = json.loads(self.request.body.decode('utf-8'))['text']
            self.finish(self.ltp._predict([text])[0])
        except Exception as e:
            self.finish(self.ltp._predict(['服务器遇到错误！'])[0])

    def options(self):
        pass


class Server(object):
    def __init__(self, path: str = 'small', batch_size: int = 50, device: str = None, onnx: bool = False):
        self.ltp = LTP(path=path, device=device)
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
            batch_dep = self.ltp.dep(hidden, fast=False)
            batch_sdp = self.ltp.sdp(hidden, mode='mix')

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

    def serve(self, port: int = 5000, n_process: int = None):
        if n_process is None:
            n_process = 1 if sys.platform == 'win32' else 8

        fmt = LogFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', color=True)
        root_logger = logging.getLogger()

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('server.log')

        console_handler.setFormatter(fmt)
        file_handler.setFormatter(fmt)

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        app_log.setLevel(logging.INFO)
        gen_log.setLevel(logging.INFO)
        access_log.setLevel(logging.INFO)

        app_log.info("Model is loading...")
        app_log.info("Model Has Been Loaded!")

        app = Application([
            (r"/.*", LTPHandler, dict(ltp=self))
        ])

        server = HTTPServer(app)
        server.bind(port)
        server.start(n_process)
        ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    Fire(Server)
