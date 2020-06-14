#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

"""
IOBES 标签列表如下：

B，即Begin，表示开始
I，即Intermediate，表示中间
E，即End，表示结尾
S，即Single，表示单个字符
O，即Other，表示其他，用于标记无关字符

子集

IOB1: 标签I用于文本块中的字符，标签O用于文本块之外的字符，标签B用于在该文本块前面接续则一个同类型的文本块情况下的第一个字符。
IOB2: 每个文本块都以标签B开始，除此之外，跟IOB1一样。
IOE1: 标签I用于独立文本块中，标签E仅用于同类型文本块连续的情况，假如有两个同类型的文本块，那么标签E会被打在第一个文本块的最后一个字符。
IOE2: 每个文本块都以标签E结尾，无论该文本块有多少个字符，除此之外，跟IOE1一样。
START/END （也叫SBEIO、IOBES）: 包含了全部的5种标签，文本块由单个字符组成的时候，使用S标签来表示，由一个以上的字符组成时，首字符总是使用B标签，尾字符总是使用E标签，中间的字符使用I标签。
IO: 只使用I和O标签，显然，如果文本中有连续的同种类型实体的文本块，使用该标签方案不能够区分这种情况。
其中最常用的是IOB2、IOBS、IOBES。
"""
from itertools import chain
from typing import List

from ltp.core import Registrable


class Representation(metaclass=Registrable):
    def encode(self, text, tags=None):
        pass

    def decode(self, tags, text: List[str] = None):
        pass


# 这里暂时没有检查标签序列一致性
class IOBES(Representation):
    B: str = 'B'  # Begin        表示开始
    I: str = 'I'  # Intermediate 表示中间
    E: str = 'E'  # End          表示结尾
    S: str = 'S'  # Single       表示单个字符
    O: str = 'O'  # Other        表示其他，用于标记无关字符

    def encode(self, words: List[str], tags: List[str] = None):
        res = []
        # tags == None
        if tags is None:
            for word in words:
                word_len = len(word)
                if word_len == 1:
                    res.append(self.S)
                elif word_len == 2:
                    res.extend([self.B, self.E])
                else:
                    res.extend([self.B] + [self.I] * (word_len - 2) + [self.E])
        else:
            tags = ['-' + tag if tag != self.O else None for tag in tags]
            for word, tag in zip(words, tags):
                word_len = len(word)
                if tag is None:
                    res.extend([self.O] * word_len)
                elif word_len == 1:
                    res.append(self.S + tag)
                elif word_len == 2:
                    res.extend([self.B + tag, self.E + tag])
                else:
                    res.extend([self.B + tag] + [self.I + tag] * (word_len - 2) + [self.E + tag])

        return res

    def decode(self, tags: List[str], text: str = None):
        res_words = []
        res_tags = []

        if text == None:
            for tag in tags:
                iobes, i_tag = tag.split('-', 1)
                if iobes == self.I or iobes == self.E:
                    pass
                elif iobes == self.S or iobes == self.B:
                    res_tags.append(i_tag)
                else:  # Others
                    res_tags.append(iobes)
            return res_tags
        else:
            for tag, char in zip(tags, text):
                try:
                    iobes, i_tag = tag.split('-', 1)
                except Exception:
                    iobes = tag
                    i_tag = ''
                if iobes == self.I or iobes == self.E:
                    res_words[-1] += char
                elif iobes == self.S or iobes == self.B:
                    res_words.append(char)
                    res_tags.append(i_tag)
                else:  # Others
                    res_words.append(char)
                    res_tags.append(iobes)
            return res_words, res_tags


class IOB2(IOBES):
    def encode(self, words: List[str], tags: List[str] = None):
        res = []
        # tags == None
        if tags is None:
            return list(chain(*(["B"] + ["I"] * (len(word) - 1) for word in words)))
        else:
            tags = ['-' + tag if tag != self.O else None for tag in tags]
            for word, tag in zip(words, tags):
                word_len = len(word)
                if tag is None:
                    res.extend([self.O] * word_len)
                else:
                    res.extend([self.B + tag] + [self.I + tag] * (word_len - 1))
        return res
