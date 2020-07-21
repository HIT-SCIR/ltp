#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: jeffrey

import os
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

    def maximum_forward_matching(self, text: str):
        maximum_matching_pos = []
        start = 0
        text_len = len(text.strip())
        while start < text_len:
            candidate = None
            for end in range(1, self.max_window + 1):
                if start + end - 1 < text_len and self[text[start:end]]:
                    candidate = (start, start + end)
                if end == self.max_window:
                    if candidate:
                        maximum_matching_pos.append(candidate)
                        start = candidate[1] - 1
                        break
                elif start + end - 1 >= text_len:
                    if candidate:
                        maximum_matching_pos.append(candidate)
                        start = candidate[1] - 1
                    break
            start = start + 1
        return maximum_matching_pos
