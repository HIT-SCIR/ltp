快速上手
========

快速安装
-----------

安装LTP是非常简单的，使用Pip安装只需要：

.. code-block:: sh

    pip install ltp

分句
-----------------
使用LTP分句只需要调用ltp.sent_split函数

.. code-block:: python

    from ltp4 import LTP
    ltp = LTP()
    sent_list = ltp.sent_split(inputs=["他叫汤姆去拿外衣。", "汤姆生病了。他去了医院。"],
                               flag="all", limit=512)

    # sent_list=["他叫汤姆去拿外衣。",
    #            "汤姆生病了。",
    #            "他去了医院。"]

分词
------------------

使用LTP分词非常简单，下面是一个简短的例子：

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    segment, _ = ltp.seg(["他叫汤姆去拿外衣。"])
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]


词性标注
------------------

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    pos = ltp.pos(hidden)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [['r', 'v', 'nh', 'v', 'v', 'n', 'wp']]

命名实体识别
------------------


.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    ner = ltp.ner(hidden)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [[('Nh', 2, 2)]]

    tag, start, end = ner[0][0]
    print(tag,":", "".join(seg[0][start:end + 1]))]
    # Nh : 汤姆



语义角色标注
------------------

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    srl = ltp.srl(hidden)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [
    #     [
    #         [],                                                # 他
    #         [('ARG0', 0, 0), ('ARG1', 2, 2), ('ARG2', 3, 5)],  # 叫 -> [ARG0: 他, ARG1: 汤姆, ARG2: 拿外衣]
    #         [],                                                # 汤姆
    #         [],                                                # 去
    #         [('ARG0', 2, 2), ('ARG1', 5, 5)],                  # 拿 -> [ARG0: 汤姆, ARG1: 外衣]
    #         [],                                                # 外衣
    #         []                                                 # 。
    #     ]
    # ]
    srl = ltp.srl(hidden, keep_empty=False)
    # [
    #     [
    #         (1, [('ARG0', 0, 0), ('ARG1', 2, 2), ('ARG2', 3, 5)]), # 叫 -> [ARG0: 他, ARG1: 汤姆, ARG2: 拿外衣]
    #         (4, [('ARG0', 2, 2), ('ARG1', 5, 5)])                  # 拿 -> [ARG0: 汤姆, ARG1: 外衣]
    #     ]
    # ]



依存句法分析
------------------

需要注意的是，在依存句法当中，虚节点ROOT占据了0位置，因此节点的下标从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    dep = ltp.dep(hidden)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [
    #     [
    #         (1, 2, 'SBV'),
    #         (2, 0, 'HED'),    # 叫 --|HED|--> ROOT
    #         (3, 2, 'DBL'),
    #         (4, 2, 'VOB'),
    #         (5, 4, 'COO'),
    #         (6, 5, 'VOB'),
    #         (7, 2, 'WP')
    #     ]
    # ]



语义依存分析(树)
------------------

与依存句法类似的，这里的下标也是从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    sdp = ltp.sdp(hidden, graph=False)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [
    #     [
    #         (1, 2, 'Agt'),
    #         (2, 0, 'Root'),   # 叫 --|Root|--> ROOT
    #         (3, 2, 'Datv'),
    #         (4, 2, 'eEfft'),
    #         (5, 4, 'eEfft'),
    #         (6, 5, 'Pat'),
    #         (7, 2, 'mPunc')
    #     ]
    # ]


语义依存分析(图)
------------------

与依存句法类似的，这里的下标也是从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    sdp = ltp.sdp(hidden, graph=True)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [
    #     [
    #         (1, 2, 'Agt'),
    #         (2, 0, 'Root'),   # 叫 --|Root|--> ROOT
    #         (3, 2, 'Datv'),
    #         (3, 4, 'Agt'),
    #         (3, 5, 'Agt'),
    #         (4, 2, 'eEfft'),
    #         (5, 4, 'eEfft'),
    #         (6, 5, 'Pat'),
    #         (7, 2, 'mPunc')
    #     ]
    # ]
