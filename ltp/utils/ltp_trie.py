#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: jeffrey
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from typing import List

import pygtrie


class Trie(pygtrie.CharTrie):
    def __init__(self, max_window=4, init_path: str = None):
        super(Trie, self).__init__()
        self._is_init = False
        self.max_window = max_window
        self.init(init_path)

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value: bool):
        if value == True:
            self._is_init = True

    def init(self, init_path: str = None, max_window=None):

        self.max_window = max_window or self.max_window

        if init_path and os.path.exists(init_path):
            self.is_init = True
            with open(init_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        self[line] = True

    def add_words(self, words):
        self.is_init = True
        if isinstance(words, str):
            self[words] = True
        if isinstance(words, list):
            for word in words:
                self[word] = True

    def maximum_forward_matching(self, text: List[str]):
        maximum_matching_pos = []
        text_len = len(text)
        for start in range(text_len - 1):
            candidate = None
            for end in range(start + 1, min(text_len, start + self.max_window + 1)):
                if self.get("".join(text[start:end]), False):
                    candidate = (start, end)
            if candidate:
                maximum_matching_pos.append(candidate)
        return maximum_matching_pos
