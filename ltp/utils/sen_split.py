#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     : 2020/6/16 4:38 PM
# @Author   : jeffrey
# @File     : sen_split.py
# @Software : PyCharm

import re


def split_sentence(document, flag="all", limit=512):
    """

    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"cn" 中文标点分句，"en" 英文标点分句
        limit: 限制最大长度为512Byte

    Returns: Type:list

    """
    sen_list = []
    try:
        if flag == "cn":
            document = re.sub('([。？！。！？]+|\\…{1,2}|\\.{3,6})(?P<quotation_mark>[”’"\'])',
                              r'\g<quotation_mark>\n', document)  # 特殊引号
            document = re.sub('\\…{1,2}|\\.{3,6}', r"\n", document)  # 中文省略号
            document = re.sub('[。？！。！？]+', r"\n", document)  # 单字符断句符

        elif flag == "en":
            document = re.sub('([?!\\.]+|\\.{3,6})(?P<quotation_mark>["\'])',
                              r'\g<quotation_mark>\n', document)  # 特殊引号
            document = re.sub('[\\.?!]+', r"\n", document)  # 英文单字符断句符

        else:
            document = re.sub('([。？！。！？\\.!?]+|\\…{1,2}|\\.{3,6})(?P<quotation_mark>[”’"\'])',
                              r'\g<quotation_mark>\n', document)  # 特殊引号
            document = re.sub('(\\…{1,2}|\\.{3,6})', r"\n", document)  # 中文省略号
            document = re.sub('[。？！。！？\\.?!]+', r"\n", document)  # 单字符断句符

        sen_list_ori = document.split("\n")
        sen_list = []
        for sen in sen_list_ori:
            sen = re.sub("\\s+", "", sen)  # 去掉空白
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


# if __name__ == "__main__":
#     document = '我们是中国人.我们在这里...今天北京天气很好?？是吗!我说："你能不能快点做好...？"'
#     print(split_sentence(document, flag="all"))