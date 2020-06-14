#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os

from tqdm import tqdm

import re
from . import Dataset
from ltp.data.example import Example
from ltp.data.utils import iter_lines


class LineDataset(Dataset, alias="line"):
    """
    Dataset 注册名: line
    每一行一个句子，词语之间按空格分割，下划线分割词语与词性，例子如下::

        上海_NR	浦东_NR	开发_NN	与_CC	法制_NN	建设_NN	同步_VV

    或者使用其他字符分割(由split参数控制)

        上海/NR	浦东/NR	开发/NN	与/CC	法制/NN	建设/NN	同步/VV

    """

    def __init__(self, path, file, fields, split='\0', maxsplit=0, silent=True, proxy_property=None,
                 line_split=None, line_strip=None, **kwargs):
        filename = os.path.join(path, file)
        examples = []
        split_regex = re.compile(split)
        fields_slices = [idx for idx, field in enumerate(fields) if field is not None]
        if proxy_property is not None:
            field_map = {field[0]: idx for idx, field in enumerate(fields)}
            for proxy, source in proxy_property.items():
                fields_slices[field_map[proxy]] = fields_slices[field_map[source]]

        for line_num, line in tqdm(list(iter_lines(filename, line_split, line_strip))):
            if len(line) == 0:
                continue
            data = [list(item) for item in zip(*(split_regex.split(item, maxsplit) for item in line))]
            data = [data[field_slice] for field_slice in fields_slices]
            try:
                examples.append(Example.fromlist(data, fields))
            except Exception as e:
                if not silent:
                    print(line_num, e, line)

        super(LineDataset, self).__init__(examples, fields, **kwargs)


from ltp.utils import deprecated


@deprecated(info='CTB数据集已经被line代替')
def CTBDataset(path, fields, silent=True, proxy_property=None, split=None, strip=None, **kwargs):
    return LineDataset(path=path, fields=fields, split='\0', silent=silent, proxy_property=proxy_property,
                       line_split=split, line_strip=strip, **kwargs)


Dataset.weak_register('CTB', CTBDataset)
