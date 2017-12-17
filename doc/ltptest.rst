使用ltp_test和xxx_cmdline
=====================

一般来讲，基于统计机器学习方法构建的自然语言处理工具通常包括两部分，即：算法逻辑以及模型。模型从数据中学习而得，通常保存在文件中以持久化；而算法逻辑则与程序对应。

.. _ltpmodel-reference-label:

LTP模型文件
-----------

LTP模型文件可以从以下渠道获取:

* `百度云 <http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569>`_ 当前模型版本3.3.1

LTP提供的模型包括:

+----------------------------+------------------------------+
| 模型名                     | 说明                         |
+============================+==============================+
| :file:`cws.model`          | 分词模型，单文件             |
+----------------------------+------------------------------+
| :file:`pos.model`          | 词性标注模型，单文件         |
+----------------------------+------------------------------+
| :file:`ner.model`          | 命名实体识别模型，单文件     |
+----------------------------+------------------------------+
| :file:`parser.model`       | 依存句法分析模型，单文件     |
+----------------------------+------------------------------+
| :file:`srl_data/`          | 语义角色标注模型，多文件     |
+----------------------------+------------------------------+


ltp_test主程序
--------------

:file:`ltp_test` 是一个整合LTP中各模块的命令行工具。它完成加载模型，依照指定方法执行分析的功能。:file:`ltp_test` 加载的模型通过配置文件指定。编译后运行::

    $ ./bin/ltp_test
    ltp_test in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    The console application for Language Technology Platform.
    
    usage: ./ltp_test <options>
    
    options:
      --threads arg           The number of threads [default=1].
      --last-stage arg        The last stage of analysis. This option can be used 
                              when the user onlywants to perform early stage 
                              analysis, like only segment without postagging.value 
                              includes:
                              - ws: Chinese word segmentation
                              - pos: Part of speech tagging
                              - ner: Named entity recognization
                              - dp: Dependency parsing
                              - srl: Semantic role labeling (equals to all)
                              - all: The whole pipeline [default]
      --input arg             The path to the input file.
      --segmentor-model arg   The path to the segment model 
                              [default=ltp_data/cws.model].
      --segmentor-lexicon arg The path to the external lexicon in segmentor 
                              [optional].
      --postagger-model arg   The path to the postag model 
                              [default=ltp_data/pos.model].
      --postagger-lexicon arg The path to the external lexicon in postagger 
                              [optional].
      --ner-model arg         The path to the NER model [default=ltp_data/ner.model
                              ].
      --parser-model arg      The path to the parser model 
                              [default=ltp_data/parser.model].
      --srl-data arg          The path to the SRL model directory 
                              [default=ltp_data/srl_data/].
      --debug-level arg       The debug level.
      -h [ --help ]           Show help information


ltp_test通过命令行参数指定分析任务与模型路径。其中，

* segmentor-model：指定分词模型
* segmentor-lexicon：指定分词词典路径
* postagger-model：指定词性标注模型
* postagger-lexicon：指定词性标注词典路径
* parser-model：指定依存句法分析模型
* ner-model：指定命名实体识别模型
* srl-data：指定语言角色标注模型
* threads：指定线程数
* input：指定输入文件，如果输入文件未指定或打开失败，将使用标准输入
* last-stage：指定分析的最终步骤。这一参数将在 :ref:`pipeline-reference-label` 中详细说明

分析结果以xml格式显示在stdout中。关于xml如何表示分析结果，请参考理解 :ref:`ltml-reference-label` 一节。

.. _pipeline-reference-label:

Pipeline与last-stage参数
~~~~~~~~~~~~~~~~~~~~~~~~

分词、词性标注、句法分析一系列任务之间存在依赖关系。举例来讲，对于词性标注，必须在分词结果之上进行才有意义。LTP中提供的5种分析之间的依赖关系如下所示：

+--------------+------+---------+
| 任务         | 标记 | 依赖    |
+==============+======+=========+
| 分词         | ws   | 无      |
+--------------+------+---------+
| 词性标注     | pos  | ws      |
+--------------+------+---------+
| 依存句法分析 | dp   | pos     |
+--------------+------+---------+
| 命名实体识别 | ner   | pos    |
+--------------+------+---------+
| 语义角色标注 | srl  | dp, ner |
+--------------+------+---------+

默认情况下，LTP将进行至语义角色标注的分析。但是，对于一部分用户，某些分析并不必要。举例来讲，如果用户只需进行词性标注，则ltp_test的pipeline分析只需进行到pos，`last-stage`用来指明分析的最后状态。同时，如果`last-stage`指定为pos，句法分析、命名实体识别和语义角色标注的模型将不被加载。

注： last-stage: 特别的，可以使用 "|" 来分割多个最终目标。例如需要ner和parser就设置为 "ner|dp"。

.. _xxxcmdline-reference-label:

xxx_cmdline
-----------
除了 :code:`ltp_test` 将全部分析器整合起来，LTP也提供拆分各模块单独分析的命令行程序。他们包括：

* :file:`cws_cmdline` ：分词命令行
* :file:`pos_cmdline` ：词性标注命令行
* :file:`par_cmdline` ：句法分析命令行
* :file:`ner_cmdline` ：命名实体识别命令行

xxx_cmdline的主要目标是提供不同于xml，同时可自由组合的语言分析模块。举在已经切分好的结果上做词性标注为例。::

    $ cat input 
    这 是 测试 文本 。
    $ cat input | ./bin/pos_cmdline --postagger-model ./ltp_data/pos.model 
    TRACE: Model is loaded
    TRACE: Running 1 thread(s)
    WARN: Cann't open file! use stdin instead.
    这_r    是_v    测试_v  文本_n  。_wp
    TRACE: consume 0.000832081 seconds.

关于各模块的用法，与ltp_test基本类似。细节请参考 :code:`xxx_cmdline -h`。

细节
----

.. _ltprestrict-reference-label:

长度限制
~~~~~~~~

为了防止输入过长句子对稳定性造成影响。ltp限制用户输入字数少于1024字，分词结果少于256词。

.. _ltpexlex-reference-label:

外部词典
~~~~~~~~

分词外部词典样例如下所示。每个制定一个词。例如 ::

    苯并芘
    亚硝酸盐

词性标注外部词典样例如下所示。每行指定一个词，第一列指定单词，第二列之后指定该词的候选词性（可以有多项，每一项占一列），列与列之间用空格区分。 ::

	雷人 v a
	】 wp

ltp项目中可以使用外部词典的模块包括

* :file:`./bin/ltp_test` ：分词、词性外部词典
* :file:`./bin/ltp_server` ：分词、词性外部词典
* :file:`./bin/examples/cws_cmdline` ：分词外部词典
* :file:`./bin/examples/pos_cmdline` ：词性外部词典
* :file:`./tools/train/otcws` ：分词外部词典
* :file:`./tools/train/otpos` ：词性外部词典

Window动态链接库
~~~~~~~~~~~~~~~~

在Window下首次运行LTP会提示找不到动态链接库，这时请将编译产生的lib/\*.dll拷贝到bin/Release/下，即可正常运行。

编码以及显示
~~~~~~~~~~~~

LTP的所有模型文件均使用UTF8 [#f1]_ 编码训练，故请确保待分析文本的编码为UTF8格式。

兼容性测试
----------

当前版本LTP在以下一些平台和编译器环境下通过测试

+-----------------+---------+---------+----------+----------+------------+----------+----------+
| 系统            | 编译器  | 版本    | ltp_test | 训练套件 | ltp_server | 单元测试 | 模型加载 |
+=================+=========+=========+==========+==========+============+==========+==========+
| Linux (64bit)   | gnu-c++ | 4.4     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | gnu-c++ | 4.6     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | gnu-c++ | 4.7     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | gnu-c++ | 4.8     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | gnu-c++ | 4.9     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | gnu-c++ | 5.3     | 支持     | 支持     | 支持       | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | clang   | 3.4     | 支持     | 支持     | 支持       | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | clang   | 3.5     | 支持     | 支持     | 支持       | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Linux (64bit)   | clang   | 3.6     | 支持     | 支持     | 支持       | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Windows (64bit) | MSVC    | 18/vs13 | 支持     | 支持     | 不支持     | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Windows (64bit) | MSVC    | 19/vs15 | 支持     | 支持     | 不支持     | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| Cygwin (64bit)  | gnu-c++ | 4.8     | 支持     | 支持     | 支持       | 支持     | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| mingw (64bit)   | gnu-c++ | 4.7     | 支持     | 支持     | 不支持     | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+
| mingw (64bit)   | gnu-c++ | 5.1     | 支持     | 支持     | 不支持     | 不支持   | 通过     |
+-----------------+---------+---------+----------+----------+------------+----------+----------+

.. rubric:: 注

.. [#f1] 由于Windows终端采用gbk编码显示，运行 :file:`ltp_test` 后会在终端输出乱码。您可以将标准输出重定向到文件，以UTF8方式查看文件，就可以解决乱码的问题。
