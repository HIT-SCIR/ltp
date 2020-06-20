#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author: jeffrey
# Changed:
#   sen -> sent / auto strip: ylfeng


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

        sent_list_ori = document.splitlines()
        sent_list = []
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list
