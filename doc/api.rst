使用编程接口
=========

如果要使用编程接口，您需要自己在开发环境中从源代码编译出相关的链接库，编译方法可以参考 :ref:`install-label` 一节

下面的文档将介绍使用LTP编译产生的静态链接库编写程序的方法。

使用动态库
-----------

如果您之前对C++使用动态库并不熟悉，可以 `参考这里 <http://msdn.microsoft.com/zh-cn/library/ms235636.aspx>`_ 。

下面以一个分词程序 :file:`cws.cpp` 来说明调用动态库的具体过程。

首先，编写调用动态库的程序，注意需要包含相应的头文件，代码如下::

	#include <iostream>
	#include <string>
	#include "segment_dll.h"

	int main(int argc, char * argv[]) {
	    if (argc < 2) {
	        std::cerr << "cws [model path]" << std::endl;
	        return 1;
	    }

	    void * engine = segmentor_create_segmentor(argv[1]);//分词接口，初始化分词器
	    if (!engine) {
	        return -1;
	    }
	    std::vector<std::string> words;
	    int len = segmentor_segment(engine,
	            "爱上一匹野马，可我的家里没有草原。", words);//分词接口，对句子分词。
	    for (int i = 0; i < len; ++ i) {
	        std::cout << words[i] << "|";
	    }
	    std::cout << std::endl;
	    segmentor_release_segmentor(engine);//分词接口，释放分词器
	    return 0;
	}

接下来，编译程序，需要加入头文件和动态库的路径。

下面给出Linux、Windows两个平台下的编译示例。

在 Windows (MSVC) 下使用动态库
~~~~~~~~~~~~~~~~~~

1. 添加头文件路径

    右键工程->Configuration properties->c/c++->general->additional include directories

    .. image:: http://p5xnn6ehz.bkt.clouddn.com/ltp-doc-img-3.6.1-1.png

2. 添加动态库路径

    右键工程->Configuration properties->linker->general->additional library directories

    .. image:: http://p5xnn6ehz.bkt.clouddn.com/ltp-doc-img-3.6.1-2.png

3. 导入所需的动态库

    右键工程->properties->linker->input->additional additional dependencies

    .. image:: http://p5xnn6ehz.bkt.clouddn.com/ltp-doc-img-3.6.1-3.png

4. 最后，Build工程即可。

在 Linux 下使用动态库
~~~~~~~

假定您下载并将LTP放置于 :file:`/path/to/your/ltp-project` 目录下,那么编译命令例如下::

    g++ -o cws cws.cpp -I /path/to/your/ltp-project/include/ -I /path/to/your/ltp-project/thirdparty/boost/include -Wl,-dn -L /path/to/your/ltp-project/lib/ -lsegmentor -lboost_regex -Wl,-dy

分词接口
--------

分词主要提供三个接口：

.. cpp:function:: void * segmentor_create_segmentor(const char * path, const char * lexicon_path)

    功能：

    读取模型文件，初始化分词器。

    参数：

    +---------------------------+------------------------------------------------------------+
    | 参数名                    | 参数描述                                                   |
    +===========================+============================================================+
    | const char * path         | 指定模型文件的路径                                         |
    +---------------------------+------------------------------------------------------------+
    | const char * lexicon_path | 指定外部词典路径。如果lexicon_path为NULL，则不加载外部词典 |
    +---------------------------+------------------------------------------------------------+

    返回值：

    返回一个指向分词器的指针。

.. cpp:function:: int segmentor_release_segmentor(void * segmentor)

    功能：

    释放模型文件，销毁分词器。

    参数：

    +---------------------------+------------------------------------------------------------+
    | 参数名                    | 参数描述                                                   |
    +===========================+============================================================+
    | void * segmentor          | 待销毁分词器的指针                                         |
    +---------------------------+------------------------------------------------------------+

    返回值：

    销毁成功时返回0，否则返回-1

.. cpp:function:: int segmentor_segment(void * segmentor, const std::string & line, std::vector<std::string> & words)

    功能：

    调用分词接口。

    参数：

    +----------------------------------+------------------------------------------------------------+
    | 参数名                           | 参数描述                                                   |
    +==================================+============================================================+
    | void * segmentor                 | 分词器的指针                                               |
    +----------------------------------+------------------------------------------------------------+
    | const std::string & line         | 待分词句子                                                 |
    +----------------------------------+------------------------------------------------------------+
    | std::vector<std::string> & words | 结果分词序列                                               |
    +----------------------------------+------------------------------------------------------------+

    返回值：

    返回结果中词的个数。

示例程序
~~~~~~~~~

一个简单的示例程序可以说明分词接口的用法::

	#include <iostream>
	#include <string>
	#include "segment_dll.h"

	int main(int argc, char * argv[]) {
	    if (argc < 2) {
	        std::cerr << "cws [model path]" << std::endl;
	        return 1;
	    }

	    void * engine = segmentor_create_segmentor(argv[1]);
	    if (!engine) {
	        return -1;
	    }
	    std::vector<std::string> words;
	    int len = segmentor_segment(engine,
	            "爱上一匹野马，可我的家里没有草原。", words);
	    for (int i = 0; i < len; ++ i) {
	        std::cout << words[i] << "|";
	    }
	    std::cout << std::endl;
	    segmentor_release_segmentor(engine);
	    return 0;
	}

示例程序通过命令行参数指定模型文件路径。第11行加载模型文件，并将分词器指针存储在engine中。第16行运行分词逻辑，并将结果存储在名为words的std::vector<std::string>中。第22行释放分词模型。

调用分词接口的程序在编译的时，需要链接segmentor.a(MSVC下需链接segmentor.lib)。

词性标注接口
--------------

词性标注主要提供三个接口

.. cpp:function:: void * postagger_create_postagger(const char * path, const char * lexicon_file)

    功能：

    读取模型文件，初始化词性标注器

    参数：

    +----------------------------------+--------------------------------------------------------------------+
    | 参数名                           | 参数描述                                                           |
    +==================================+====================================================================+
    | const char * path                | 词性标注模型路径                                                   |
    +----------------------------------+--------------------------------------------------------------------+
    | const char * lexicon_file        | 指定词性标注外部词典路径。如果lexicon_file为NULL，则不加载外部词典 |
    +----------------------------------+--------------------------------------------------------------------+

    lexicon_file参数指定的外部词典文件样例如下所示。每行指定一个词，第一列指定单词，第二列之后指定该词的候选词性（可以有多项，每一项占一列），列与列之间用空格区分::

        雷人 v a
        】 wp

    返回值：

    返回一个指向词性标注器的指针。

.. cpp:function:: int postagger_release_postagger(void * postagger)

    功能：

    释放模型文件，销毁分词器。

    参数：

    +----------------------------------+--------------------------------------------------------------------+
    | 参数名                           | 参数描述                                                           |
    +==================================+====================================================================+
    | void * postagger                 | 待销毁的词性标注器的指针                                           |
    +----------------------------------+--------------------------------------------------------------------+

    返回值：

    销毁成功时返回0，否则返回-1

.. cpp:function:: int postagger_postag(void * postagger, const std::vector<std::string> & words, std::vector<std::string> & tags)

    功能：

    调用词性标注接口

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | void * postagger                       | 词性标注器的指针                                                   |
    +----------------------------------------+--------------------------------------------------------------------+
    | const std::vector<std::string> & words | 待标注的词序列                                                     |
    +----------------------------------------+--------------------------------------------------------------------+
    | std::vector<std::string> & tags        | 词性标注结果，序列中的第i个元素是第i个词的词性                     |
    +----------------------------------------+--------------------------------------------------------------------+


    返回值：

    返回结果中词的个数

示例程序
~~~~~~~~~

一个简单的示例程序可以说明词性标注接口的用法::

	#include <iostream>
	#include <vector>
	#include "postag_dll.h"

	int main(int argc, char * argv[]) {
	    if (argc < 1) {
	        return -1;
	    }

	    void * engine = postagger_create_postagger(argv[1]);
	    if (!engine) {
	        return -1;
	    }

	    std::vector<std::string> words;

	    words.push_back("我");
	    words.push_back("是");
	    words.push_back("中国人");

	    std::vector<std::string> tags;

	    postagger_postag(engine, words, tags);

	    for (int i = 0; i < tags.size(); ++ i) {
	        std::cout << words[i] << "/" << tags[i];
	        if (i == tags.size() - 1) std::cout << std::endl;
	        else std::cout << " ";

	    }

	    postagger_release_postagger(engine);
	    return 0;
	}

示例程序通过命令行参数指定模型文件路径。第11行加载模型文件，并将词性标注器指针存储在engine中。第18至20行构造分词序列，第24行运行词性标注逻辑，并将结果存储在名为tags的std::vector<std::string>中。第33行释放分词模型。

调用词性标注接口的程序在编译的时，需要链接postagger.a(MSVC下需链接postagger.lib)。

命名实体识别接口
------------------

命名实体识别主要提供三个接口：

.. cpp:function:: void * ner_create_recognizer(const char * path)

    功能：

    读取模型文件，初始化命名实体识别器

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | const char * path                      | 命名实体识别模型路径                                               |
    +----------------------------------------+--------------------------------------------------------------------+

    返回值：

    返回一个指向词性标注器的指针。

.. cpp:function:: int ner_release_recognizer(void * recognizer)

    功能：

    释放模型文件，销毁命名实体识别器。

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | void * recognizer                      | 待销毁的命名实体识别器的指针                                       |
    +----------------------------------------+--------------------------------------------------------------------+

    返回值：

    销毁成功时返回0，否则返回-1

.. cpp:function:: int ner_recognize(void * recognizer, const std::vector<std::string> & words, const std::vector<std::string> & postags, std::vector<std::string> tags)

    功能：

    调用命名实体识别接口

    参数：

    +------------------------------------------+----------------------------------------------------------------------------------------+
    | 参数名                                   | 参数描述                                                                               |
    +==========================================+========================================================================================+
    | void * recognizer                        | 命名实体识别器的指针                                                                   |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | const std::vector<std::string> & words   | 待识别的词序列                                                                         |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | const std::vector<std::string> & postags | 待识别的词的词性序列                                                                   |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | std::vector<std::string> & tags          | | 命名实体识别结果，                                                                   |
    |                                          | | 命名实体识别的结果为O时表示这个词不是命名实体，                                      |
    |                                          | | 否则为{POS}-{TYPE}形式的标记，POS代表这个词在命名实体中的位置，TYPE表示命名实体类型  |
    +------------------------------------------+----------------------------------------------------------------------------------------+

    返回值：

    返回结果中词的个数

示例程序
~~~~~~~~~

一个简单的示例程序可以说明命名实体识别接口的用法::


	#include <iostream>
	#include <vector>
	#include "ner_dll.h"

	int main(int argc, char * argv[]) {
	    if (argc < 2) {
	        std::cerr << "usage: ./ner [model_path]" << std::endl;
	        return -1;
	    }

	    void * engine = ner_create_recognizer(argv[1]);
	    if (!engine) {
	        std::cerr << "failed to load model" << std::endl;
	        return -1;
	    }

	    std::vector<std::string> words;
	    std::vector<std::string> postags;

	    words.push_back("中国");    postags.push_back("ns");
	    words.push_back("国际");    postags.push_back("n");
	    words.push_back("广播");    postags.push_back("n");
	    words.push_back("电台");    postags.push_back("n");
	    words.push_back("创办");    postags.push_back("v");
	    words.push_back("于");      postags.push_back("p");
	    words.push_back("1941年");  postags.push_back("m");
	    words.push_back("12月");    postags.push_back("m");
	    words.push_back("3日");     postags.push_back("m");
	    words.push_back("。");      postags.push_back("wp");

	    std::vector<std::string>    tags;

	    ner_recognize(engine, words, postags, tags);

	    for (int i = 0; i < tags.size(); ++ i) {
	        std::cout << words[i] << "\t" << postags[i] << "\t" << tags[i] << std::endl;
	    }

	    ner_release_recognizer(engine);
	    return 0;
	}

示例程序通过命令行参数指定模型文件路径。第11行加载模型文件，并将命名实体识别器指针存储在engine中。第21至30行构造分词序列words和词性标注序列postags，第34行运行词性标注逻辑，并将结果存储在名为tags的std::vector<std::string>中。第40行释放分词模型。

调用命名实体识别接口的程序在编译的时，需要链接ner.a（MSVC下需链接ner.lib）。

依存句法分析接口
-----------------

依存句法分析主要提供三个接口：

.. cpp:function:: void * parser_create_parser(const char * path)

    功能：

    读取模型文件，初始化依存句法分析器

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | const char * path                      | 依存句法分析模型路径                                               |
    +----------------------------------------+--------------------------------------------------------------------+

    返回值：

    返回一个指向依存句法分析器的指针。

.. cpp:function:: int parser_release_parser(void * parser)

    功能：

    释放模型文件，销毁依存句法分析器。

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | void * parser                          | 待销毁的依存句法分析器的指针                                       |
    +----------------------------------------+--------------------------------------------------------------------+

    返回值：

    销毁成功时返回0，否则返回-1

.. cpp:function:: int parser_parse(void * parser, const std::vector<std::string> & words, const std::vector<std::string> & postagger, std::vector<int> & heads, std::vector<std::string> & deprels)

    功能：

    调用依存句法分析接口

    参数：

    +------------------------------------------+--------------------------------------------------------------------+
    | 参数名                                   | 参数描述                                                           |
    +==========================================+====================================================================+
    | void * parser                            | 依存句法分析器的指针                                               |
    +------------------------------------------+--------------------------------------------------------------------+
    | const std::vector<std::string> & words   | 待分析的词序列                                                     |
    +------------------------------------------+--------------------------------------------------------------------+
    | const std::vector<std::string> & postags | 待分析的词的词性序列                                               |
    +------------------------------------------+--------------------------------------------------------------------+
    | std::vector<int> & heads                 | 结果依存弧，heads[i]代表第i个词的父亲节点的编号                    |
    +------------------------------------------+--------------------------------------------------------------------+
    | std::vector<std::string> & deprels       | 结果依存弧关系类型                                                 |
    +------------------------------------------+--------------------------------------------------------------------+

    返回值：

    返回结果中词的个数

示例程序
~~~~~~~~~

一个简单的示例程序可以说明依存句法分析接口的用法::

	#include <iostream>
	#include <vector>
	#include "parser_dll.h"

	int main(int argc, char * argv[]) {
	    if (argc < 2) {
	        return -1;
	    }

	    void * engine = parser_create_parser(argv[1]);
	    if (!engine) {
	        return -1;
	    }

	    std::vector<std::string> words;
	    std::vector<std::string> postags;

	    words.push_back("一把手");      postags.push_back("n");
	    words.push_back("亲自");        postags.push_back("d");
	    words.push_back("过问");        postags.push_back("v");
	    words.push_back("。");          postags.push_back("wp");

	    std::vector<int>            heads;
	    std::vector<std::string>    deprels;

	    parser_parse(engine, words, postags, heads, deprels);

	    for (int i = 0; i < heads.size(); ++ i) {
	        std::cout << words[i] << "\t" << postags[i] << "\t"
	            << heads[i] << "\t" << deprels[i] << std::endl;
	    }

	    parser_release_parser(engine);
	    return 0;
	}

示例程序通过命令行参数指定模型文件路径。第11行加载模型文件，并将依存句法分析器指针存储在engine中。第19至22行构造分词序列words和词性标注序列postags，第27行运行词性标注逻辑，并将依存弧关系存储在heads中，将依存弧关系类型存储在deprels中。第34行释放依存句法分析模型。

调用依存句法分析接口的程序在编译的时，需要链接parser.a(MSVC下需链接parser.lib)。

语义角色标注接口
-------------------

语义角色标注主要提供三个接口：

.. cpp:function:: int SRL_loadResource(const std::string& ConfigDir)

    功能：

    读取模型文件，初始化语义角色标注器

    参数：

    +----------------------------------------+--------------------------------------------------------------------+
    | 参数名                                 | 参数描述                                                           |
    +========================================+====================================================================+
    | const std::string& ConfigDir           | 语义角色标注模型文件夹所在路径                                     |
    +----------------------------------------+--------------------------------------------------------------------+

    返回值：

    返回一个指向词性标注器的指针。

.. cpp:function:: int SRL_ReleaseResource()

    功能：

    释放模型文件，销毁命名实体识别器。

    返回值：

    销毁成功时返回0，否则返回-1

.. cpp:function:: int DoSRL(const std::vector<std::string> & words, \
                            const std::vector<std::string> & POS, \
                            const std::vector<std::string>& NEs, \
                            const std::vector< std::pair<int, std::string> >& parse, \
                            std::vector< \
                                std::pair< \
                                    int, \
                                    std::vector< \
                                        std::pair< \
                                            std::string, \
                                            std::pair<int, int> \
                                        > \
                                    > \
                                > \
                            >& SRLResult)

    功能：

    调用命名实体识别接口

    参数：

    +------------------------------------------+----------------------------------------------------------------------------------------+
    | 参数名                                   | 参数描述                                                                               |
    +==========================================+========================================================================================+
    | const std::vector<std::string> & words   | 输入的词序列                                                                           |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | const std::vector<std::string> & postags | 输入的词性序列                                                                         |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | const std::vector<std::string> & nes     | 输入的命名实体序列                                                                     |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | | const std::vector<                     | | 输入的依存句法结果                                                                   |
    | |     std::pair<int, std::string>        | | 依存句法结果表示为长度为句子长的序列                                                 |
    | | > & parse                              | | 序列中每个元素由两个成员组成，分别表示这个词的父节点的编号 [#f1]_ 和依存关系类型     |
    +------------------------------------------+----------------------------------------------------------------------------------------+
    | | std::vector<                           | | 输出的语义角色标注结果                                                               |
    | |     std::pair<                         | | 语义角色标注结果表示为一个句子中谓词个数的序列                                       |
    | |         int,                           | | 序列中每个谓词有两个成员组成，第一个成员表示谓词的下标，第二个成员是一个列表         |
    | |         std::vector<                   | | 列表中每个元素表示与这个谓词对应的论元                                               |
    | |             std::pair<                 | | 每个论元由两个成员组成：                                                             |
    | |                 std::string,           | | 第一个成员代表这个论元的语义角色类型，                                               |
    | |                 std::pair<int, int>    | | 第二个成员代表这个论元的管辖范围，表示成一个二元组                                   |
    | |             >                          |                                                                                        |
    | |         >                              |                                                                                        |
    | |     >                                  |                                                                                        |
    | | >                                      |                                                                                        |
    +------------------------------------------+----------------------------------------------------------------------------------------+

    返回值：

    返回结果中词的个数

.. rubric:: 注

.. [#f1] 编号从0记起
