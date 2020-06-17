#!usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     : 2020/6/16 10:40 AM
# @Author   : jeffrey
# @File     : sen_split.py
# @Software : PyCharm


import re
import itertools


def split_sentence(document, flag="all", limit=512):
    """

    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"ch" 中文标点分句，"en" 英文标点分句
        limit: 限制最大长度为512个字符

    Returns: Type:list

    """
    sen_list = []
    try:
        if flag == "ch":
            document = re.sub('(?P<quotation_mark>([。？！。！？…]+(?![”’"\'])))', r'\g<quotation_mark>\n',
                              document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！。！？]+|…{1,2})[”’"\'])', r'\g<quotation_mark>\n',
                              document)  # 特殊引号

        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([\\.?!]+(?![”’"\'])))', r'\g<quotation_mark>\n',
                              document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!\\.]+["\']))',
                              r'\g<quotation_mark>\n', document)  # 特殊引号

        else:
            document = re.sub('(?P<quotation_mark>([。？！。！？…\\.?!]+(?![”’"\'])))', r'\g<quotation_mark>\n',
                              document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！。！？\\.!?]+|\\…{1,2})[”’"\']))',
                              r'\g<quotation_mark>\n', document)  # 特殊引号

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


# if __name__ == "__main__":
#     document = ['我们是 中国人.我们\r在这里...今天北京天气…很好?？是吗!我说："你能不能快点做好...？"',
#                 '你好啊。我们爱你。你来这里吧."']
#     documents = [split_sentence(x, flag="all") for x in document]
#     document = list(itertools.chain(*documents))
#     print(document)
