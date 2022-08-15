#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import codecs


def iter_raw_lines(filename: str, strip=None, skip: str = None):
    line_num = 0
    with codecs.open(filename, encoding="utf-8") as file:
        while True:
            line = file.readline()
            line_num += 1
            if skip is not None and line.startswith(skip):
                continue
            if not line:  # EOF
                yield line_num, ""  # 输出空行，简化上层逻辑
                break
            line = line.strip(strip)
            yield line_num, line


def iter_lines(filename: str, split=None, strip=None, skip: str = None):
    for line_num, raw_line in iter_raw_lines(filename=filename, strip=strip, skip=skip):
        if not raw_line:  # end of a sentence
            yield line_num, []  # 输出空行
        else:
            yield line_num, raw_line.split(split)


def iter_blocks(filename: str, split=None, strip=None, skip="#"):
    rows = []
    for line_num, line_features in iter_lines(filename, split=split, strip=strip, skip=skip):
        if len(line_features):
            rows.append(line_features)
        else:
            if len(rows):
                yield line_num, rows
                rows = []
