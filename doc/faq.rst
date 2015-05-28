常见问题
========

1. *“我在本地使用ltp的时候出现 [ERROR] … in LTP::wordseg, failed to load segmentor resource 是怎么回事儿？”*

这个提示的含义是模型加载失败。

**可能原因一** ：模型不存在

如果您没有下载模型，请参考 :ref:`ltpmodel-reference-label` 下载部署模型

**可能原因二** ：模型与ltp版本不对应

ltp在加载模型是会检查模型签名和当前版本号，所以请确定您使用的模型与您使用的ltp版本号对应。
ltp版本号可以用 :code:`./bin/ltp_test --help` 查看。
模型版本号可以通过 :code:`ltp_data/version` 查看。

**未知原因**

LTP模型使用二进制文件存储。
由于 :code:`unsigned long long` 在不同编译器下长度不同，可能存在加载出错。
对于这种问题，请在我们项目的issue tracker https://github.com/HIT-SCIR/ltp/issues 里面反馈问题。
在提交issue时，请将您的编译器情况、系统情况（32bit/64bit等）反隐给我们。

2. *“我使用分词词典了，但是为什么某些词典词还是被切开了”*

ltp的分词模块 **并非采用词典匹配的策略** ，外部词典以特征方式加入机器学习算法，并不能保证所有的词都是按照词典里的方式进行切分。
如果要完全按照词典匹配的方式切词，您可以尝试对切词结果进行后处理。

3. *“可不可以把Java/Python调用LTP，或切词原理文档， 发我一份吗？”*

使用Java或Python调用ltp请参考ltp4j和pyltp这两个项目以及文档中 :ref:`otherlang-reference-label` 部分。

4. *“调用ltp_server时为什会出现400错误？”*

**可能原因一** ：句子过长

为了保证处理效率，ltp对于输入句子长度进行了限制。现在的限制是最大句子长度 *1024字* ，切词结果最多 *256词* 。

**可能原因二** ：编码错误

ltp只接受UTF8编码输入。如果您的输入是GBK编码，请转为 *UTF8无bom编码* 。

5. *“我有很长很长的句子，如何才能打破1024字/256词的限制？”*

**方案一** ：使用 :ref:`xxxcmdline-reference-label`

您可以使用 :code:`xxx_cmdline` 作为替代。:code:`xxx_cmdline` 的分词模块、词性标注模块和命名实体识别模块是没有长度限制的。
句法分析模块限制长度为1024词。

**方案二** ：修改 :file:`src/ltp/Ltp.h`

修改 :file:`src/ltp/Ltp.h`中 :code:`#define MAX_SENTENCE_LEN 1024` 和 :code:`#define MAX_WORDS_NUM 256` 两个宏，重新编译。

6. *我没找到想要的答案，请问哪里能获得有关ltp的帮助*

您可以在我们的Google group https://groups.google.com/forum/#!forum/ltp-cloud 中发帖提问。
提问前，请再次确认您的问题没有现成的答案。并在提问时保持礼貌风度。
有关提问的艺术，请参考池建强老师的博客 http://macshuo.com/?p=367 。

