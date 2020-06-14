#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os

import numpy
from tqdm import tqdm

from . import Dataset
from ltp.data.example import Example
from ltp.data.utils import iter_blocks
from ltp.utils import expand_bio


class CorpusDataset(Dataset, alias="Corpus"):
    """
    Dataset 注册名: Corpus
    Conll 文本标准，不同Field之间使用 Tab 分割，不同句子之间有一空行，例子如下::

        科学	_	O	B-ARG0	B-ARG0	O
        发展观	_	O	I-ARG0	I-ARG0	O
        绝对	_	O	B-ARGM-ADV	B-ARGM-ADV	O
        不	_	O	B-ARGM-ADV	B-ARGM-ADV	O
    """

    def __init__(self, path, file, fields, multi_field=None,
                 split=None, strip=None, proxy_property=None, **kwargs):
        filename = os.path.join(path, file)
        examples = list(self.iter(filename, fields, multi_field, split, strip, proxy_property))
        super(CorpusDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def build_slice(fields, multi_field=None, ignore_fields=None):
        used_fields = []
        field_row = []

        ignored = 0
        if ignore_fields is None:
            ignore_fields = {}

        for idx, field in enumerate(fields):
            if field is None:
                continue
            used_fields.append(field)
            if field[0] in ignore_fields:
                ignored += 1
                field_row.append(-1)
                continue
            field_row.append(idx - ignored)
        if multi_field is None:
            return used_fields, field_row

        field_names = [field[0] if field is not None else str(idx) for idx, field in enumerate(fields)]
        if multi_field not in field_names:
            return used_fields, field_row

        mf_tag_idx = field_names.index(multi_field)
        mf_tag_id = field_row[mf_tag_idx]
        if mf_tag_idx == len(field_names) - len(ignore_fields) - 1:  # 是最后一个
            field_slices = field_row[:mf_tag_idx] + [slice(mf_tag_id, None)]
        else:
            raise NotImplementedError("Multifield must be last row")

        return used_fields, [None if field_name in ignore_fields else field_slices[idx]
                             for idx, (field_name, field) in enumerate(used_fields)]

    def iter(self, filename: str, fields, multi_field: str = None, split=None, strip=None, proxy_property: dict = None):
        fields, fields_slices = self.build_slice(fields, multi_field, proxy_property)
        if proxy_property is not None:
            field_map = {field[0]: idx for idx, field in enumerate(fields)}
            for proxy, source in proxy_property.items():
                if proxy in field_map:
                    fields_slices[field_map[proxy]] = fields_slices[field_map[source]]

        for line_num, block in tqdm(list(iter_blocks(filename, split, strip))):
            values = [list(value) for value in zip(*block)]
            values = [values[field_slice] for field_slice in fields_slices]
            processed, more = self.post_fn(values)

            if more:
                for values in processed:
                    try:
                        yield Example.fromlist(values, fields)
                    except Exception as e:
                        print(line_num, e)
            else:
                try:
                    yield Example.fromlist(processed, fields)
                except Exception as e:
                    print(line_num, e)

    def post_fn(self, input):
        return input, False
