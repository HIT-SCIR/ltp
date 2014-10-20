#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

toc = ["简介",
        "开始使用ltp",
        "使用ltp_test以及模型",
        "编程接口",
        "使用其他语言调用ltp",
        "使用ltp_server",
        "实现原理与性能",
        "使用训练套件",
        "发表论文",
        "附录"]

if __name__=="__main__":
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print >> sys.stderr, "argument is not a directory"
        sys.exit(1)

    print """LTP使用文档v3.0
===============

#### 更新信息

* 刘一佳 << yjliu@ir.hit.edu.cn >> 2014年6月14日，增加使用其他语言调用ltp一节
* 牛国成 << gcniu@ir.hit.edu.cn >> 2014年5月10日，增加词性词典相关文档
* 韩冰 << bhan@ir.hit.edu.cn >> 2014年1月16日，增加模型裁剪相关文档
* 刘一佳 << yjliu@ir.hit.edu.cn >> 2013年7月17日，创建文档

版权所有：哈尔滨工业大学社会计算与信息检索研究中心
"""

    print """## 目录"""

    for title in toc:
        print "* [%s](#%s)" % (title, title)

    print
    for title in toc:
        try:
            fp=open(os.path.join(sys.argv[1], title + ".md"), "r")
        except:
            continue

        print "# %s" % title
        print fp.read().strip()
        print
