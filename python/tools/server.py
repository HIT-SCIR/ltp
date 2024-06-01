#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Modify from https://raw.githubusercontent.com/HIT-SCIR/ltp/4.1/tools/server.py
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
"""
LTP Server 是对 LTP 的一个简单包装，依赖于 tornado，使用方式如下：

.. code-block:: bash

    pip install ltp, tornado
    python tools/server.py serve
"""
import sys
import json
import logging
from typing import List

import torch

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
            print(self.request.body.decode('utf-8'))
            text = json.loads(self.request.body.decode('utf-8'))['text']
            #print(text)
            result = self.ltp._predict([text])
            #print(result)
            self.finish(result)
        except Exception as e:
            self.finish(self.ltp._predict(['服务器遇到错误！'])[0])

    def options(self):
        pass


class Server(object):
    def __init__(self, path: str = 'base', batch_size: int = 50, device: str = None, onnx: bool = False):
        # 2024/6/1 7:9:45 adapt for "ltp==4.2.13"
        self.ltp = LTP('LTP/base')
        # 将模型移动到 GPU 上
        if torch.cuda.is_available():
            # ltp.cuda()
            self.ltp.to("cuda")

    def _predict(self, sentences: List[str]):
        #result = []
        output = self.ltp.pipeline(sentences, tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])


        # https://github.com/HIT-SCIR/ltp/blob/main/python/interface/docs/quickstart.rst
        # 需要注意的是，在依存句法当中，虚节点ROOT占据了0位置，因此节点的下标从1开始。
        id = 0
        offset = 0
        words = []
        for word, pos, parent, relation in \
                zip(output.cws[0], output.pos[0], output.dep[0]['head'], output.dep[0]['label']):
            #print([id, word, pos, parent, relation])
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


        for token_srl in output.srl[0]:
            for argument in token_srl['arguments']:
                #print(token_srl['index'], token_srl['predicate'], argument)
                text = argument[1]
                start = argument[2]
                offset = words[start]['offset']
                words[token_srl['index']]['roles'].append({
                    'text': text,
                    'offset': offset,
                    'length': len(text),
                    'type': argument[0]
                })


        start = 0
        for end, label in \
                zip(output.sdp[0]['head'], output.sdp[0]['label']):
            words[start]['parents'].append({'parent': end - 1, 'relate': label})
            start = start + 1


        nes = []
        for role, text, start, end in output.ner[0]:
            nes.append({
                'text': text,
                'offset': start,
                'ne': role.lower(),
                'length': len(text)
            })


        result = {
            'text': sentences[0],
            'nes': nes,
            'words': words
        }

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

        #app_log.info("Model is loading...")
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
