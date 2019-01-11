使用ltp_server
==============

注意
----

ltp_server只提供Linux版本

本文档中提到的LTP Server与语言云服务不同。语言云建立在LTP Server之上，并封装了一层REST API接口。语言云API(ltp-cloud-api)的请求方式与LTP Server不同。


搭建LTP Server
---------------

LTP Server在轻量级服务器程序mongoose基础上开发。在编译LTP源码之后，运行`./bin/ltp_server`就可以启动LTP Server。::

    ltp_server in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    The HTTP server frontend for Language Technology Platform.
    
    usage: ./ltp_server <options>
    
    options:
      --port arg              The port number [default=12345].
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
      --log-level arg         The log level:
                              - 0: TRACE level
                              - 1: DEBUG level
                              - 2: INFO level [default]
                              
      -h [ --help ]           Show help information


其中较为重要的参数包括：

- last-stage: 特别的，可以使用 "|" 来分割多个最终目标。例如需要ner和parser就设置为 "ner|dp"。
- port：指定LTP server监听的端口
- threads：指定LTP server运行的线程数，线程数影响并发的处理能力
- log-level：指定日志级别，TRACE级别最低，显示日志信息最详细。INFO级别最高，显示日志最粗略。WARN与ERROR级日志默认显示。

其余参数用以指定模型路径，具体含义与`ltp_test`相同。

请求LTP Server
---------------

原理
~~~~~

在ltp web service中，client与server之间采用http协议通信。client以post方式提交的数据到server，server将数据以xml的方式返回给client。

client提交的post请求主要有以下几个字段。

+--------+--------------------------------------------------------------------------------------------------------------------------------------+
| 字段名 | 含义                                                                                                                                 |
+========+======================================================================================================================================+
| s      | 输入字符串，在xml选项x为n的时候，代表输入句子；为y时代表输入xml                                                                      |
+--------+--------------------------------------------------------------------------------------------------------------------------------------+
| x      | 用以指明是否使用xml                                                                                                                  |
+--------+--------------------------------------------------------------------------------------------------------------------------------------+
| t      | 用以指明分析目标，t可以为分词（ws）,词性标注（pos），命名实体识别（ner），依存句法分析（dp），语义角色标注（srl）或者全部任务（all） |
+--------+--------------------------------------------------------------------------------------------------------------------------------------+
| f      | 用以指明返回格式。f=xml或f=json（默认）  |
+--------+--------------------------------------------------------------------------------------------------------------------------------------+

.. _ltml-reference-label:

数据表示
~~~~~~~~~~

LTP 数据表示标准称为LTML。下面是LTML的一个简单例子::

	<?xml version="1.0" encoding="utf-8" ?>
	<xml4nlp>
	    <note sent="y" word="y" pos="y" ne="y" parser="y" wsd="y" srl="y" />
	    <doc>
	        <para id="0">
	            <sent id="0" cont="我们都是中国人">
	                <word id="0" cont="我们" pos="r" ne="O" parent="2" relate="SBV" />
	                <word id="1" cont="都" pos="d" ne="O" parent="2" relate="ADV" />
	                <word id="2" cont="是"  pos="v" ne="O" parent="-1" relate="HED">
	                    <arg id="0" type="A0" beg="0" end="0" />
	                    <arg id="1" type="AM-ADV" beg="1" end="1" />
	                </word>
	                <word id="3" cont="中国" pos="ns" ne="S-Ns" parent="4" relate="ATT" />
	                <word id="4" cont="人" pos="n" ne="O" parent="2" relate="VOB" />
	            </sent>
	        </para>
	    </doc>
	</xml4nlp>

LTML 标准要求如下：

结点标签分别为 xml4nlp, note, doc, para, sent, word, arg 共七种结点标签：

1. xml4nlp 为根结点，无任何属性值；

2. note 为标记结点，具有的属性分别为：sent, word, pos, ne, parser, srl；
   分别代表分句，分词，词性标注，命名实体识别，依存句法分析，词义消歧，语义角色标注；
   值为"n"，表明未做，值为"y"则表示完成，如pos="y"，表示已经完成了词性标注；

3. doc 为篇章结点，以段落为单位包含文本内容；无任何属性值；

4. para 为段落结点，需含id 属性，其值从0 开始；

5. sent 为句子结点，需含属性为id，cont；
   
   a) id 为段落中句子序号，其值从0 开始；
   b) cont 为句子内容；
   
6. word 为分词结点，需含属性为id, cont；
   
   a) id 为句子中的词的序号，其值从0 开始，
   b) cont为分词内容；可选属性为 pos, ne, parent, relate；
      
      I) pos 的内容为词性标注内容；
      II) ne 为命名实体内容；
      III) parent 与relate 成对出现，parent 为依存句法分析的父亲结点id 号，relate 为相对应的关系；
      
7. arg 为语义角色信息结点，任何一个谓词都会带有若干个该结点；其属性为id, type, beg，end；
   
   a) id 为序号，从0 开始；
   b) type 代表角色名称；
   c) beg 为开始的词序号，end 为结束的序号；

各结点及属性的逻辑关系说明如下：

1. 各结点层次关系可以从图中清楚获得，凡带有id 属性的结点是可以包含多个；
2. 如果sent="n"即未完成分句，则不应包含sent 及其下结点；
3. 如果sent="y" word="n"即完成分句，未完成分词，则不应包含word 及其下结点；
4. 其它情况均是在sent="y" word="y"的情况下：

   a) 如果 pos="y"则分词结点中必须包含pos 属性；
   b) 如果 ne="y"则分词结点中必须包含ne 属性；
   c) 如果 parser="y"则分词结点中必须包含parent 及relate 属性；
   d) 如果 srl="y"则凡是谓词(predicate)的分词会包含若干个arg 结点；

示例程序
~~~~~~~~~~

下面这个python程序例子显示如何向LTP Server发起http请求，并获得返回结果::

    # -*- coding: utf-8 -*-
    #!/usr/bin/env python
    import urllib, urllib2

    uri_base = "http://127.0.0.1:12345/ltp"

    data = {
        's': '我爱北京天安门',
        'x': 'n',
        't': 'all'}

    request = urllib2.Request(uri_base)
    params = urllib.urlencode(data)
    response = urllib2.urlopen(request, params)
    content = response.read().strip()
    print content

错误返回
~~~~~~~~

如果请求有不符合格式要求，LTP Server会返回400错误。下面的表格显示了LTP Server返回的错误类型以及原因。

+-------+----------------------+---------------------------------------------------+
| code  | reason               | 解释                                              |
+=======+======================+===================================================+
| 400   | EMPTY SENTENCE       | 输入句子为空                                      |
+-------+----------------------+---------------------------------------------------+
| 400   | ENCODING NOT IN UTF8 | 输入句子非UTF8编码                                |
+-------+----------------------+---------------------------------------------------+
| 400   | SENTENCE TOO LONG    | 输入句子不符合 :ref:`ltprestrict-reference-label` |
+-------+----------------------+---------------------------------------------------+
| 400   | BAD XML FORMAT       | 输入句子不符合LTML格式                            |
+-------+----------------------+---------------------------------------------------+

当前版本服务性能
----------------

版本：3.3.0

测试使用Xeon(R) CPU E5-2620 0 @ 2.00GHz，4线程，请求时间：3分钟，测试脚本使用pylot 1.26。

Number of agents = 10

+------------+----------------------+----------------------+
| Last Stage | Response Time (secs) | Throughput (req/sec) |
+============+======================+======================+
| ws         | 0.010                | 643.308              |
+------------+----------------------+----------------------+
| pos        | 0.012                | 743.809              |
+------------+----------------------+----------------------+
| dp         | 0.016                | 574.785              |
+------------+----------------------+----------------------+
| ne	     | 0.014                | 673.661              |
+------------+----------------------+----------------------+
| srl/all    | 0.036                | 266.094              |
+------------+----------------------+----------------------+
