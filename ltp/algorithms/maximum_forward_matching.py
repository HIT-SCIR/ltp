#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: jeffrey
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from typing import List

import pygtrie


class Trie(pygtrie.CharTrie):
    __is_init: bool
    __max_window: int
    __min_start: int

    def __init__(self, max_window=4, init_path: str = None):
        super(Trie, self).__init__()
        self.__is_init = False
        self.__min_start = max_window
        self.__max_window = max_window
        self.init(init_path)

    @property
    def is_init(self):
        return self.__is_init

    @is_init.setter
    def is_init(self, value: bool):
        if value:
            self.__is_init = True

    @property
    def max_window(self):
        return self.__max_window

    @max_window.setter
    def max_window(self, value: int):
        if isinstance(value, int) and value > 0:
            self.__max_window = value
            self.__min_start = min(self.__min_start, self.__max_window)

    @property
    def min_start(self):
        return self.__min_start

    @min_start.setter
    def min_start(self, value: int):
        if isinstance(value, int) and value > 0:
            self.__min_start = min(value, self.__min_start, self.__max_window)

    def init(self, init_path: str = None, max_window=None):
        self.max_window = max_window
        if init_path and os.path.exists(init_path):
            self.is_init = True
            with open(init_path, "r", encoding="utf-8") as f:
                for word in f.readlines():
                    word = word.strip()
                    if word:
                        self[word] = True
                        self.min_start = len(word)

    def add_words(self, words):
        self.is_init = True
        if isinstance(words, str):
            self[words] = True
            self.min_start = len(words)
        if isinstance(words, list):
            for word in words:
                self[word] = True
                self.min_start = len(word)

    def maximum_forward_matching(self, text: List[str], preffix: List[bool]):
        maximum_matching_pos = []
        text_len = len(text)
        start = 0
        while start <= text_len - 1:
            if preffix[start]:
                candidate = None
                min_len = start + self.min_start - 1
                max_len = min(text_len, start + self.max_window)
                for end in range(max_len, min_len, -1):
                    if self.get("".join(text[start:end]), False):
                        candidate = (start, end)
                        break
                if candidate:
                    start += candidate[1]
                    maximum_matching_pos.append(candidate)
                else:
                    start += 1

        return maximum_matching_pos
