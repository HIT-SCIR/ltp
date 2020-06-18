#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author: jeffrey


import re
import itertools


def split_sentence(document: str, flag: str = "all", limit: int = 512):
    """

    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 限制最大长度为512个字符

    Returns: Type:list

    """
    sen_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！。！？…](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！。！？]|…{1,2})[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号

        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([\\.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!\\.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号

        else:
            document = re.sub('(?P<quotation_mark>([。？！。！？…\\.?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                              document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！。！？\\.!?]|\\…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              document)  # 特殊引号

        sen_list_ori = document.splitlines()
        sen_list = []
        for sen in sen_list_ori:
            # sen = re.sub("\\s+", "", sen)  # 去掉空白
            if not sen:
                continue
            else:
                while len(sen) > limit:
                    temp = sen[0:limit]
                    sen_list.append(temp)
                    sen = sen[limit:]
                sen_list.append(sen)
    except:
        sen_list.clear()
        sen_list.append(document)
    return sen_list
