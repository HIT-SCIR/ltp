LTP使用文档v3.0
===============

#### 作者

* 刘一佳 << yjliu@ir.hit.edu.cn>> 2013年7月17日创建文档

版权所有：哈尔滨工业大学社会计算与信息检索研究中心

## 目录


# 简介

语言技术平台（Language Technology Platform，LTP）是哈工大社会计算与信息检索研究中心历时十年开发的一整套中文语言处理系统。LTP制定了基于XML的语言处理结果表示，并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块（包括词法、句法、语义等6项中文处理核心技术），以及基于动态链接库（Dynamic Link Library, DLL）的应用程序接口，可视化工具，并且能够以网络服务（Web Service）的形式进行使用。

从2006年9月5日开始该平台对外免费共享目标代码，截止目前，已经有国内外400多家研究单位共享了LTP，也有国内外多家商业公司购买了LTP，用于实际的商业项目中。2010年12月获得中国中文信息学会颁发的行业最高奖项：“钱伟长中文信息处理科学技术奖”一等奖。

2011年6月1日，为了与业界同行共同研究和开发中文信息处理核心技术，我中心正式将LTP开源。

# 开始使用LTP

如果你是第一次使用LTP，不妨花一些时间了解LTP能帮你做什么。

LTP提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。从应用角度来看，LTP为用户提供了下列组件：

* 针对单一自然语言处理任务，生成统计机器学习模型的工具
* 针对单一自然语言处理任务，调用模型进行分析的编程接口
* 使用流水线方式将各个分析工具结合起来，形成一套统一的中文自然语言处理系统
* 系统可调用的，用于中文语言处理的模型文件
* 针对单一自然语言处理任务，基于云端的编程接口

如果你的公司需要一套高性能的中文语言分析工具以处理海量的文本，或者你的在研究工作建立在一系列底层中文自然语言处理任务之上，或者你想将自己的科研成果与前沿先进工作进行对比，LTP都可能是你的选择。

# 如何安装LTP

下面的文档将介绍如何安装LTP

## 获得LTP

作为安装的第一步，你需要获得LTP。LTP包括两部分，分别是项目源码和编译好的模型文件。你可以从以下链接获得最新的LTP项目源码。

* Github项目托管：[https://github.com/HIT-SCIR/ltp/releases](https://github.com/HIT-SCIR/ltp/releases)

同时，你可以从以下一些地方获得LTP的模型。

* 

## 安装CMake

LTP使用编译工具CMake构建项目。在安装LTP之前，你需要首先安装CMake。CMake的网站在[这里](http://www.cmake.org)。如果你是Windows用户，请下载CMake的二进制安装包；如果你是Linux，Mac OS或Cygwin的用户，可以通过编译源码的方式安装CMake，当然，你也可以使用Linux的软件源来安装。

## Windows(MSVC)编译

第一步：构建VC Project

在项目文件夹下新建一个名为build的文件夹，使用CMake Gui，在source code中填入项目文件夹，在binaries中填入build文件夹。然后Configure -> Generate。

或者在命令行build 路径下运行

	cmake ..

第二步：编译

构建后得到ALL_BUILD、RUN_TESTS、ZERO_CHECK三个VC Project。使用VS打开ALL_BUILD项目，选择Release方式构建项目。

（注：由于boost::multi_array与VS2010不兼容，并不能使用Debug方式构建项目。）

## Linux，Mac OSX和Cygwin编译

Linux、Mac OSX和Cygwin的用户，可以直接在项目根目录下使用命令


	./configure
	./make


进行编译。

## 简单地试用

编译成功后，会在./bin文件夹下生成如下一些二进制程序：

| 程序名 | 说明 |
| ------ | ---- |
| ltp_test | LTP调用程序 |
| ltp_server* | LTP Server程序 |

在lib文件夹下生成以下一些静态链接库

| 程序名 | 说明 |
| ------ | ---- |
| splitsnt.lib | 分句lib库 |
| segmentor.lib | 分词lib库 |
| postagger.lib | 词性标注lib库 |
| parser.lib | 依存句法分析lib库 |
| ner.lib | 命名实体识别lib库 |
| srl.lib | 语义角色标注lib库 |

同时，在tools/train文件夹下会产生如下一些二进制程序：

| 程序名 | 说明 |
| ------ | ---- |
| otcws | 分词的训练和测试套件 |
| otpos | 词性标注的训练和测试套件 |
| lgdpj | 依存句法分析训练和测试套件 |
| maxent* | 最大熵工具包，用于训练命名实体识别和语义角色标注模型 |
| SRLExtract* | 语义角色标注训练程序 |
| SRLGetInstance* | |

（注*：在window版本中ltp_server、Maxent、SRLExtract、SRLGetInstance并不被编译。）

# 使用ltp_test以及模型

一般来讲，基于统计机器学习方法构建的自然语言处理工具通常包括两部分，即：算法逻辑以及模型。模型从数据中学习而得，通常保存在文件中以持久化；而算法逻辑则与程序对应。

ltp提供一整套算法逻辑以及模型，其中的模型包括：

| 模型名 | 说明 |
| ----- | ---- |
| cws.model | 分词模型，单文件 |
| pos.model | 词性标注模型，单文件 |
| ner_data/ | 命名实体识别模型，多文件 |
| parser.model | 依存句法分析模型，单文件 |
| srl_data/ | 语义角色标注模型，多文件 |

ltp_test是一个整合ltp中各模块的命令行工具。他完成加载模型，依照指定方法执行分析的功能。ltp_test加载的模型通过配置文件指定。配置文件的样例如下：

    segmentor-model = new_ltp_data/cws.model
	postagger-model = new_ltp_data/pos.model
	parser-model = new_ltp_data/parser.model
	ner-data = ltp_data/ne_data
	srl-data = ltp_data/srl_data

其中，

* segmentor-model项指定分词模型
* postagger-model项指定词性标注模型
* parser-model项指定依存句法分析模型
* ner-data项指定命名实体识别模型
* srl-data项指定语言角色标注模型

ltp_test的使用方法如下：

	./bin/ltp_test [配置文件] [分析目标] [待分析文件]

分析结果以xml格式显示在stdout中。关于xml如何表示分析结果，请参考理解Web Service Client结果一节。

# 编程接口

下面的文档将介绍使用LTP编译产生的静态链接库编写程序的方法。

（注：2.30以后，LTP的所有模型文件均使用UTF8编码训练，故请确保待分析文本的编码为UTF8格式）

## 分词接口

分词主要提供三个接口：

* `void * segmentor_create_segmentor(const char * path, const char * lexicon_path);`

读取模型文件，初始化分词器。Path指定模型文件的路径。返回一个指向分词器的指针。lexicon_path指定外部词典路径。如果为空，则不加载外部词典。

* `int segmentor_release_segmentor(void * segmentor);`

释放模型文件，销毁分词器。传入待销毁的分词器的指针segmentor。

* `int segmentor_segment(void * segmentor, const std::string & line, std::vector<std::string> & words);`

调用分词接口。Segmentor指定分词器，line为待分词句子，结果词序列存储在words中。返回结果中词的个数。

### 示例程序

一个简单的实例程序可以说明分词接口的用法：

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

调用分词接口的程序在编译的时，需要链接segmentor.a（MSVC下需链接segmentor.lib）。

## 词性标注接口

词性标注主要提供三个接口

* `void * postagger_create_postagger(const char * path);`

读取模型文件，初始化词性标注器。Path指定模型文件的路径。返回一个指向分词器的指针。

* `int postagger_release_postagger(void * postagger);`

释放模型文件，销毁词性标注器。传入待销毁的词性标注器的指针postagger。

* `int postagger_postag(void * postagger, const std::vector< std::string > & words, std::vector<std::string> & tags);`

调用词性标注接口。指定分词器，words为待标注的词序列，标注结果存储在tags中。返回结果中词的个数。

### 示例程序

一个简单的实例程序可以说明词性标注接口的用法：

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

调用词性标注接口的程序在编译的时，需要链接postagger.a（MSVC下需链接postagger.lib）。

## 命名实体识别接口

## 依存句法分析接口

依存句法分析主要提供三个接口：

* `void * parser_create_parser(const char * path);`

读取模型文件，初始化词性标注器。Path指定模型文件的路径。返回一个指向分词器的指针。

* `int parser_release_parser(void * parser);`

释放模型文件，销毁词性标注器。传入待销毁的依存句法分析器的指针parser。

* `int parser_parse(void * parser, const std::vector< std::string > & words, const std::vector< std::string > & postags, std::vector<int> & heads, std::vector<std::string> & deprels);`

调用依存句法分析接口。指定依存句法分析器，words为待分析的词序列，postags为词序列对应的词性序列。依存句法树结构存储在heads中，依存弧关系存储在deprels中。

### 示例程序

一个简单的实例程序可以说明依存句法分析接口的用法：

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

调用依存句法分析接口的程序在编译的时，需要链接parser.a（MSVC下需链接parser.lib）。

## 语义角色标注接口

# Web Service使用方法

除C++的编程接口，LTP还提供针对自然语言处理任务，基于云端的编程接口。用户可以通过使用LTP Web Service Client调用云端服务。

* 免安装：用户只需要下载LTP Web Service客户端源代码，编译执行后即可获得分析结果，无需调用静态库或下载模型文件。
* 省硬件：LTP Web Service Client几乎可以运行于任何硬件配置的计算机上，用户不需要购买高性能的机器，即可快捷的获得分析结果。
* 跨平台跨语言：LTP Web Service客户端几乎可以运行于任何操作系统之上，无论是Windows、Linux各个发行版或者Mac OS。
* 跨编程语言：时至今日，LTP Web Service Client已经提供了包括C++，Java，C#在内的编程接口，其他语言的编程接口也在开发之中。

在运算资源有限，编程语言受限的情况下，Web Service无疑是用户使用LTP更好的选择。

## 获得Web Service Client

你可以从以下链接获得最新的LTP项目源码。

* github项目托管：[https://github.com/HIT-SCIR/ltp-service](https://github.com/HIT-SCIR/ltp-service])

## 利用Web Service Client获得分析结果

### C++

### Java

### C#

## 理解Web Service Client结果

LTP 数据表示标准称为LTML。下图是LTML的一个简单例子：

	<?xml version="1.0" encoding=" gbk " ?>
	<xml4nlp>
	    <note sent="y" word="y" pos="y" ne="y" parser="y" wsd="y" srl="y" />
	    <doc>
	        <para id="0">
	            <sent id="0" cont="我们都是中国人">
	                <word id="0" cont="我们" wsd="Aa02" wsdexp="我_我们" pos="r" ne="O" parent="2" relate="SBV" />
	                <word id="1" cont="都" wsd="Ka07" wsdexp="都_只_不止_甚至" pos="d" ne="O" parent="2" relate="ADV" />
	                <word id="2" cont="是" wsd="Ja01" wsdexp="是_当做_比作" pos="v" ne="O" parent="-1" relate="HED">
	                    <arg id="0" type="A0" beg="0" end="0" />
	                    <arg id="1" type="AM-ADV" beg="1" end="1" />
	                </word>
	                <word id="3" cont="中国" wsd="Di02" wsdexp="国家_行政区划" pos="ns" ne="S-Ns" parent="4" relate="ATT" />
	                <word id="4" cont="人" wsd="Aa01" wsdexp="人_人民_众人" pos="n" ne="O" parent="2" relate="VOB" />
	            </sent>
	        </para>
	    </doc>
	</xml4nlp>

LTML 标准要求如下：结点标签分别为 xml4nlp, note, doc, para, sent, word, arg 共七种结点标签：

1. xml4nlp 为根结点，无任何属性值；
2. note 为标记结点，具有的属性分别为：sent, word, pos, ne, parser, wsd, srl；分别代表分句，分词，词性标注，命名实体识别，依存句法分析，词义消歧，语义角色标注；值为”n”，表明未做，值为”y”则表示完成，如pos=”y”，表示已经完成了词性标注；
3. doc 为篇章结点，以段落为单位包含文本内容；无任何属性值；
4. para 为段落结点，需含id 属性，其值从0 开始；
5. sent 为句子结点，需含属性为id，cont；id 为段落中句子序号，其值从0 开始；cont 为句子内容；
6. word 为分词结点，需含属性为id, cont；id 为句子中的词的序号，其值从0 开始，cont为分词内容；可选属性为 pos, ne, wsd, wsdexp, parent, relate；pos 的内容为词性标注内容；ne 为命名实体内容；wsd 与wsdexp 成对出现，wsd 为词义消歧内容，wsdexp 为相应的解释说明；parent 与relate 成对出现，parent 为依存句法分析的父亲结点id 号，relate 为相对应的关系；
7. arg 为语义角色信息结点，任何一个谓词都会带有若干个该结点；其属性为id, type, beg，end；id 为序号，从0 开始；type 代表角色名称；beg 为开始的词序号，end 为结束的序号；

各结点及属性的逻辑关系说明如下：

1. 各结点层次关系可以从图中清楚获得，凡带有id 属性的结点是可以包含多个；
2. 如果sent=”n”即未完成分句，则不应包含sent 及其下结点；
3. 如果sent=”y” word=”n”即完成分句，未完成分词，则不应包含word 及其下结点；
4. 其它情况均是在sent=”y” word=”y”的情况下：
	1. 如果 pos=”y”则分词结点中必须包含pos 属性；
	2. 如果 ne=”y”则分词结点中必须包含ne 属性；
	3. 如果 parser=”y”则分词结点中必须包含parent 及relate 属性；
	4. 如果 wsd=”y”则分词结点中必须包含wsd 及wsdexp 属性；
	5. 如果 srl=”y”则凡是谓词（predicate）的分词会包含若干个arg 结点；

# 搭建一个私人的LTP Server

LTP Server在轻量级服务器程序mongoose基础上开发。在编译LTP源码之后，运行LTP Server就可以启动LTP Server。LTP Server启动后，将会监听12345（*）端口的HTTP请求。

（注*：如需指定监听其他端口，请在src/server/ltp_server.cpp中将宏LISTENING_PORT “12345”设置为其他整数即可。）

# 实现原理与性能

## 在线学习算法框架

在机器学习领域，在线学习(Online learning)指每次通过一个训练实例学习模型的学习方法。在线学习的目的是正确预测训练实例的标注。在线学习最重要的一个特点是，当一次预测完成时，其正确结果便被获得，这一结果可直接用来修正模型。

在自然语言处理领域，在线学习已经被广泛地应用在分词、词性标注、依存句法分析等结构化学习任务中。

## 分词模块

在LTP中，我们将分词任务建模为基于字的序列标注问题。对于输入句子的字序列，模型给句子中的每个字标注一个标识词边界的标记。在LTP中，我们采用的标记集如附录所示。

对于模型参数，我们采用在线机器学习算法框架从标注数据中学习参数。对于分词模型，我们使用的基本模型特征有：

| 类别 | 特征 |
| --- | --- |
| char-unigram | ch[-2], ch[-1], ch[0], ch[1], ch[2] |
| char-bigram | ch[-2]ch[-1], ch[-1]ch[0],ch[0]ch[1],ch[1]ch[2] |
| dulchar | ch[-1]=ch[0]? |
| dul2char | ch[-2]=ch[0]? |

同时，为了提高互联网文本特别是微博文本的处理性能。我们在分词系统中加入如下一些优化策略：

* 英文、URI一类特殊词识别规则
* 利用空格等自然标注线索
* 在统计模型中融入词典信息
* 从大规模未标注数据中统计的字间互信息、上下文丰富程度

在统计模型中融合词典的方法是将最大正向匹配得到的词特征

| 类别 | 特征 |
| --- | --- 
| begin-of-lexicon-word | ch[0] is preffix of words in lexicon? |
| middle-of-lexicon-word | ch[0] is middle of words in lexicon? |
| end-of-lexicon-word | ch[0] is suffix of words in lexicon? |

基础模型在几种数据集上的性能如下：

### 人民日报

语料信息：人民日报1998年2月-6月（后10%数据作为开发集）作为训练数据，1月作为测试数据。

* 准确率为：

| P | R | F |
|---|---|---|
| 开发集 | 0.973152 | 0.972430 | 0.972791 |
| 测试集 | 0.972316 | 0.970354 | 0.972433 |

* 运行时内存：520540/1024=508.3m
* 速度：5543456/30.598697s=176.91k/s

### CTB5

CTB5数据来源于，训练集和测试集按照官方文档中建议的划分方法划分。

* 准确率为：

| P | R | F |
|---|---|---|
|开发集 | 0.941426 | 0.937309 | 0.939363 |
|测试集 | 0.967235 | 0.973737 | 0.970475 |

* 运行时内存：141980/1024=138.65M
* 速度：50518/0.344988 s=143.00k/s

### CTB6

CTB6数据来源于，训练集和测试集按照官方文档中建议的划分方法划分。

* 准确率为：

| P | R | F |
|---|---|---|
|开发集|0.933438 | 0.940648 | 0.937029 |
|测试集|0.932683 | 0.938023 | 0.935345 |

* 运行时内存：116332/1024=113.6M
* 速度：484016/2.515181 s=187.9k/s

## 词性标注模块

与分词模块相同，我们将词性标注任务建模为基于词的序列标注问题。对于输入句子的词序列，模型给句子中的每个词标注一个标识词边界的标记。在LTP中，我们采用的北大标注集。关于北大标注集信息，请参考：

对于模型参数，我们采用在线机器学习算法框架从标注数据中学习参数。对于词性标注模型，我们使用的模型特征有：

| 类别 | 特征 |
| --- | --- |
| word-unigram |w[-2], w[-1], w[0], w[1], w[2] |
| word-bigram | w[-2]w[-1],w[-1]w[0],w[0]w[1],w[1]w[2] |
| word-trigram | w[-1]w[0]w[1] |
| last-first-character |ch[0,0]ch[0,n],ch[-1,n]ch[0,0],ch[0,-1]ch[1,0] |
| length | length |
| prefix | ch[0,0],ch[0,0:1],ch[0,0:2]|
| suffix | ch[0,n-2:n],ch[0,n-1:n],ch[0,n]|

基础模型在几种数据集上的性能如下：

### 人民日报

语料信息：人民日报1998年2月-6月（后10%数据作为开发集）作为训练数据，1月作为测试数据。

* 准确率为：

|  | P |
| --- | --- |
|开发集 | 0.979621 |
|测试集 | 0.978337 |

* 运行时内存：1732584/1024=1691.97m
* 速度：5543456/51.003626s=106.14k/s

### CTB5

CTB5数据来源于，训练集和测试集按照官方文档中建议的划分方法划分。

* 准确率为：

| | P |
| --- | --- |
|开发集 | 0.953819 |
|测试集 | 0.946179 |

* 运行时内存：356760/1024=348.40M
* 速度：50518/0.527107 s=93.59k/s

### CTB6

CTB6数据来源于，训练集和测试集按照官方文档中建议的划分方法划分。

* 准确率为：

| | P |
| --- | --- |
|开发集 | 0.939930 |
|测试集 | 0.938439 |

* 运行时内存：460116/1024=449.33M
* 速度：484016/5.735547 s=82.41k/s

## 命名实体识别模块

## 依存句法分析模块

基于图的依存分析方法由McDonald首先提出，他将依存分析问题归结为在一个有向图中寻找最大生成树（Maximum Spanning Tree）的问题。
在依存句法分析模块中，LTP分别实现了

* 一阶解码(1o)
* 二阶利用子孙信息解码(2o-sib)
* 二阶利用子孙和父子信息(2o-carreras)

三种不同的解码方式。依存句法分析模块中使用的特征请参考：

在LDC数据集上，三种不同解码方式对应的性能如下表所示。

| model | 1o | | 2o-sib | | 2o-carreras | |
| ----- | --- | ---| ----- |---| ---------- |---|
| | Uas | Las | Uas | Las | Uas | Las |
|Dev | 0.8190 | 0.7893 | 0.8501 | 0.8213 | 0.8582 | 0.8294 |
|Test | 0.8118 | 0.7813 | 0.8421 | 0.8106 | 0.8447 | 0.8138 |
|Speed | 49.4 sent./s | | 9.4 sent./s | | 3.3 sent./s |
|Mem. | 0.825g | | 1.3g | | 1.6g |

## 语义角色标注模块

# 使用训练套件

## 分词训练套件otcws用法

otcws是ltp分词模型的训练套件，用户可以使用otcws训练获得ltp的分词模型。

编译之后，在tools/train下面会产生名为otcws的二进制程序。调用方法是

	./otcws [config_file]。

otcws分别支持从人工切分数据中训练分词模型和调用分词模型对句子进行切分。人工切分的句子的样例如下：

	对外		，	他们		代表		国家		。

otcws主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示。

	[train]
	train-file = data/ctb5-train.seg
	holdout-file = data/ctb5-holdout.seg
	algorithm = pa 
	model-name = model/ctb5-seg
	max-iter = 5

其中，

* [train] 配置组指定执行训练
	* Ttain-file 配置项指定训练集文件
	* Holdout-file 配置项指定开发集文件
	* Algorithm 指定参数学习方法，现在otcws支持两种参数学习方法，分别是passive aggressive（pa）和average perceptron（ap）。
	* Model-name 指定输出模型文件名
	* Max-iter 指定最大迭代次数

测试配置的配置文件样例如下所示。

	[test]
	test-file = data/ctb5-test.seg
	model-file = model/ctb5-seg.4.model

其中，

* [test] 配置组指定执行测试
	* Test-file 指定测试文件
	* Model-file 指定模型文件位置
	
切分结果将输入到标准io中。

（*[train]与[test]两个配置组不能同时存在）

## 词性标注训练套件otpos用法

otpos是ltp分词模型的训练套件，用户可以使用otpos训练获得ltp的分词模型。

编译之后，在tools/train下面会产生名为otpos的二进制程序。调用方法是

	./otpos [config_file]

otpos分别支持从人工切分并标注词性的数据中训练词性标注模型和调用词性标注模型对切分好的句子进行词性标注。人工标注的词性标注句子样例如下：

	对外_v	，_wp	他们_r	代表_v	国家_n	。_wp

otpos主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示。

	[train]
	train-file = data/ctb5-train.pos
	holdout-file = data/ctb5-holdout.pos
	algorithm = pa
	model-name = model/ctb5-pos
	max-iter = 5

其中，

* [train] 配置组指定执行训练
	* Ttain-file 配置项指定训练集文件
	* Holdout-file 配置项指定开发集文件
	* Algorithm 指定参数学习方法，现在otcws支持两种参数学习方法，分别是passive aggressive（pa）和average perceptron（ap）。
	* Model-name 指定输出模型文件名
	* Max-iter 指定最大迭代次数

测试配置的配置文件样例如下所示。

其中，

* [test] 配置组指定执行测试
	* Test-file 指定测试文件
	* Model-file 指定模型文件位置

词性标注结果将输入到标准io中。

（*[train]与[test]两个配置组不能同时存在）

## 依存句法分析训练套件lgdpj用法

lgdpj是ltp依存句法分析模型的训练套件，用户可以使用lgdpj训练获得ltp的依存句法分析模型。

编译之后，在tools/train下面会产生名为lgdpj的二进制程序。调用方法是

	./lgdpj [config_file]。

lgdpj分别支持从人工标注依存句法的数据中训练依存句法分析模型和调用依存句法分析模型对句子进行依存句法分析。人工标注的词性标注依存句法的句子遵从conll格式，其样例如下：

	1       对外    _       v       _       _       4       ADV     _       _
	2       ，      _       wp      _       _       1       WP      _       _
	3       他们    _       r       _       _       4       SBV     _       _
	4       代表    _       v       _       _       0       HED     _       _
	5       国家    _       n       _       _       4       VOB     _       _
	6       。      _       wp      _       _       4       WP      _       _

lgdpj主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示。

	[model]
	labeled = 1
	decoder-name = 2o-carreras
	
	[feature]
	use-postag-unigram = 0
	use-dependency = 1
	use-dependency-unigram = 1
	use-dependency-bigram = 1
	use-dependency-surrounding = 1
	use-dependency-between = 1
	use-sibling = 1
	use-sibling-basic = 1
	use-sibling-linear = 1
	use-grand = 1
	use-grand-basic = 1
	use-grand-linear = 1
	
	[train]
	train-file = data/conll/ldc-train.conll
	holdout-file = data/conll/ldc-holdout.conll
	max-iter = 5 
	algorithm = pa
	model-name = model/parser/ldc-o2carreras

其中，

* [mode] 配置组中
	* labeled 表示是否使用有label的依存句法分析
	* decoder-name 表示采用的解码算法，现在lgdpj支持三种解码算法，分别是1o，2o-sib，2o-carreras
* [feature] 配置组指定使用的特征
* [train] 配置组指定执行训练
	* Train-file 配置项指定训练集文件
	* Holdout-file 配置项指定开发集文件
	* Algorithm 指定参数学习方法，现在otcws支持两种参数学习方法，分别是passive aggressive（pa）和average perceptron（ap）。
	* Model-name 指定输出模型文件名
	* Max-iter 指定最大迭代次数

测试配置的配置文件样例如下所示。

	[test]
	test-file = data/conll/ldc-test.conll
	model-file = model/parser/ldc-o2carreras.2.model

其中，

* [test] 配置组指定执行测试
	* Test-file 指定测试文件
	* Model-file 指定模型文件位置

依存句法分析结果将输入到标准io中。

# 发表论文

# 附录

## 分词标注集

| 标记 | 含义 | 举例 |
| --- | --- | --- |
| B | 词首 | __中__国 |
| I | 词中 | 哈__工__大 |
| E | 词尾 | 科__学__ |
| S | 单字成词 | 的 |
 
## 词性标注集

LTP 使用的是863 词性标注集，其各个词性含义如下表。

| Tag | Description         | Example    | Tag | Description       | Example    |
| --- | ------------------- | ---------- | --- | ----------------- | ---------- |
| a   | adjective           | 美丽       | ni  | organization name | 保险公司   |
| b   | other noun-modifier | 大型, 西式 | nl  | location noun     | 城郊       |
| c   | conjunction         | 和, 虽然   | ns  | geographical name | 北京       |
| d   | adverb              | 很         | nt  | temporal noun     | 近日, 明代 |
| e   | exclamation         | 哎         | nz  | other proper noun | 诺贝尔奖   |
| g   | morpheme            | 茨, 甥     | o   | onomatopoeia      | 哗啦       |
| h   | prefix              | 阿, 伪     | p   | preposition       | 在, 把     |
| i   | idiom               | 百花齐放   | q   | quantity          | 个         |
| j   | abbreviation        | 公检法     | r   | pronoun           | 我们       |
| k   | suffix              | 界, 率     | u   | auxiliary         | 的, 地     |
| m   | number              | 一, 第一   | v   | verb              | 跑, 学习   |
| n   | general noun        | 苹果       | wp  | punctuation       | ，。！     |
| nd  | direction noun      | 右侧       | ws  | foreign words     | CPU        |
| nh  | person name         | 杜甫, 汤姆 | x   | non-lexeme        | 萄, 翱     |
