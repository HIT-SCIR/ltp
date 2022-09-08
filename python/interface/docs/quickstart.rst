快速上手
========

快速安装
-----------

安装LTP是非常简单的，使用Pip安装只需要：

.. code-block:: sh

    pip install ltp

载入模型
--------------------------

.. code-block:: python

    from ltp import LTP
    ltp = LTP() # 默认加载 LTP/Small 模型
    # ltp = LTP(path = "LTP/base|LTP/small|LTP/tiny")

分句
--------------------------

使用LTP分句只需要调用ltp.sent_split函数

.. code-block:: python

    from ltp import StnSplit
    sents = StnSplit().sent_split(["他叫汤姆去拿外衣。", "汤姆生病了。他去了医院。"])

    # [
    #   "他叫汤姆去拿外衣。",
    #   "汤姆生病了。",
    #   "他去了医院。"
    # ]

用户自定义词典
-------------------

.. code-block:: python

    from ltp import LTP
    ltp = LTP()
    # 也可以在代码中添加自定义的词语
    ltp.add_words(word="长江大桥", freq = 2)


分词
------------------

使用LTP分词非常简单，下面是一个简短的例子：

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    words = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws"], return_dict = False)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]


词性标注
------------------

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","pos"])
    print(result.pos)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
    # [['r', 'v', 'nh', 'v', 'v', 'n', 'wp']]

命名实体识别
------------------


.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","ner"])
    print(result.ner)
    # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]



语义角色标注
------------------

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","srl"])
    print(result.srl)



依存句法分析
------------------

需要注意的是，在依存句法当中，虚节点ROOT占据了0位置，因此节点的下标从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","dep"])
    print(result.dep)



语义依存分析(树)
------------------

与依存句法类似的，这里的下标也是从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","sdp"])
    print(result.sdp)


语义依存分析(图)
------------------

与依存句法类似的，这里的下标也是从1开始。

.. code-block:: python

    from ltp import LTP

    ltp = LTP()

    result = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks = ["cws","sdpg"])
    print(result.sdpg)


LTP Server
------------------------------

LTP Server 是对 LTP 的一个简单包装，依赖于 tornado，使用方式如下：

.. code-block:: bash

    pip install ltp, tornado
    python utils/server.py serve
