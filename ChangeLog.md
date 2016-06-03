2016-04-29 v3.3.2
------------------
* [修复] 修复了 3.3.1 版本的一些 bug

2016-03-30 v3.3.1
------------------
* [修复] 修复了 3.3.0 版本模型加载的 bug
* [修复] 修复了 gcc5、mingw、VS2015 下的编译问题
* [增加] 提供 Windows 下的 `ltp_test`和 `xxx_cmdline` 二进制下载

2015-05-23
----------
* [增加] 词性标注模型加入微博数据，使得在开放域上的词性标注性能更好(+3.3 precision)
* [增加] 依存句法分析模型加入微博数据，使得在开放域上的句法分析性能更好(+3 UAS)
* [增加] 依存句法分析算法切换到transition-based neural network parser，速度从40 tokens/s提升到8000 tokens/s。同时通过加入聚类特征以及优化训练算法，（在新闻领域）准确率也得到小幅提升(+0.2 UAS)
* [增加] `ltp_test`默认支持多线程，线程数可配置。
* [增加] 新加入子模块命令行程序，`cws_cmdline`，`pos_cmdline`，`par_cmdline`，`ner_cmdline`，使用户可以很容易替换中间模块，从而实现语言分析的组合。
* [修改] 优化了训练套件的交互方式
* [增加] 添加模型验证，单元测试模块。

2014-11-16
----------
* [增加] 分词模块增量模型训练工具。使用户可以在语言技术平台基线模型的基础之上增加训练数据，从而获得特定领域性能更好的模型。
* [修改] Boost.Regex到1.56.0，由于旧版本Boost.Regex的`match_results`类存在竞争问题，这一修改修复了`multi_cws_cmdline`随机出错的问题。
* [修改] 使自动化测试脚本支持Windows运行以及多线程测试
* [修改] 将原`examples`文件夹下的示例文件转移到`test`文件夹下并纳入语言技术平台的编译链
* [测试] 新版语言技术平台通过`cygwin`编译测试
* [测试] 多线程程序`multi_ltp_test`，`multi_cws_cmdline`以及`multi_pos_cmdline`在Windows通过测试

2014-10-11
----------
* [修改] 修改`utils/template.hpp`的实现，提高40%的速度性能
* [修改] 修改`_WIN32`宏在mingw下的歧义，使得LTP在`Codeblocks - Mingw Makefile`模式下正常编译
* [修改] 修改非unix系统的编译目标，使得win32与win64都不进行ltp_server以及unittest的编译
* [增加] 自动化测试脚本

2014-01-20
----------
* 在分词、词性标注和依存句法分析模块中加入模型裁剪功能，减少了模型大小。用户可以通过配置文件里的rare-feature-threshold参数配置裁剪力度，如果rare-feature-threshold为0，则只去掉为0的特征；rare-feature-threshold大于0时将一步去掉更新次数低于阈值的特征。这一优化方法主要参考[Learning Sparser Perceptron Models](http://www.cs.bgu.ac.il/~yoavg/publications/acl2011sparse.pdf)。
* 增加了`ltp_server`在异常输入情况下返回错误代码，如果输入数据编码错误或者输入xml不符合规则，将返回400
* 修复了词性标注、命名实体识别、依存句法分析训练套件中的内存泄露问题
* 修复了语义角色标注的内存泄露问题
* 修复了词性标注、命名实体识别模型文件的错误标示符，这项修改将导致3.1.0以及之后的版本不能与3.0.x的模型兼容，请务必注意
* 修复了由boost.multi_array.views引起的MSVC下不能以Debug方式编译的问题
* 修复了由打开文件时字符串为空引起的Windows下不能正常运行的bug

2013-09-29
----------
* 解决windows编译问题
* 实现各模块多线程支持
* 新增linux下多线程LTP工具包，multi_ltp_test
* 实现服务器程序ltp_server多线程支持
* 修复4长度utf-8字符、伪标记导致%的标注结果等bug

2013-08-04
----------
* 从底层开始，实现了一套中文文本处理库
* 实现序列标注算法框架
* 在序列标注算法框架的基础上实现了分词、词性标注、命名实体识别和依存句法分析四个模块
* 在分词模块中实现了用户自定义字典的逻辑
* 在依存句法分析模块中实现了二阶解码，提高分析准确率
* 实现模型裁剪，提高内存性能

2013-04-03
----------
* 将LTP的训练模块进行封装，用户可以直接调用ltp-model脚本训练模型
* ltp_data` 可在如下地址下载: http://ir.hit.edu.cn/ltp/program/ltp_data.zip

2013-03-19
----------
* 将crf++-0.50和maxent-20061005整合到LTP中
* 将LTP中依赖的boost库提取出来，整合到LTP中。当前版本整合的boost库是1.48.0
* 当前编译LTP不需要boost库和crf++、maxent

2013-03-15
----------
* 将LTP的编译工具从automake转换到CMake

2011-06-02
----------
* 所有的模型文件存放在ltp_data.zip中，因此运行LTP时需要先将该压缩包解压至当前文件夹。

2010-02-02
----------
* 升级SRL程序以及模型文件

2009-12-18
----------
* `ltp_data` 可在如下地址下载: http://node03.gaoliqi.com/down/ltp_data.tgz


2009-10-12
----------
* 同时支持Windows和Linux


2009-06-19
----------
v2.1

1. 增加CRFWordSeg接口
2. 解决了若干svmtagger的bug
3. 解决了若干ner的bug
4. 解决LTP对文字进行修改的bug
5. 解决使用vector作为DLL接口参数类型的bug (VS2008下出错)
6. 更新LTP使用文档
7. 最新版我们只提供vs2008对应的DLL，如果希望在visual studio其他版本上运行，可以尝试安装[Microsoft Visual C++ 2008 Redistributable Package (x86)](http://www.microsoft.com/downloads/details.aspx?FamilyID=9B2DA534-3E03-4391-8A4D-074B9F2BC1BF&displaylang=en)

2008-12-30
----------
* 使用svn来管理代码、数据、文档，并且将share-package也加入到这个体系中，支持自动发布:tools/distrubition.dat
* 增加vs2008支持
* 修改文档：LTP使用文档，updates.txt等

2008-12-28
----------
* 针对LTP网站上遇到输入空文本（只包含空格，全角空格）便会重新加载资源的问题：
    * 如果处理过程中出错，不exit，只说明出错，并log
    * 在ltp_v2.0_start.py中增加文档检查是否为空文本，如果是便不能提交。
* 网站使用最新DLL: gparser
* 刘龙提出分句模块还有些不合理，准备借鉴刘龙的分句正则表达式(python)，使用boost库修改现在的模块。

2008-12-26
----------

* 将mstparser更名为gparser (graph-based parser)
* vs2008下编译运行时发现两个bug:
    1. _gparser/MyLib.cpp中my_getline(): size_t end = xx.size() - 1; 改成 int end..
    2. _gparser/DepParser.cpp中addArcFeature: inst.postag[0]越界
    3. 修改_gparser/DepParser.cpp中fillInstance(): 增加postag, feat的resize
* 没有更新网上的共享v2.0


2008-12-21
----------

* 修改mstparser输出 wp->n 的情况，保证任何词的父亲不会是wp
* 删除sds, text-classify, cr相关代码，数据；保留xml4nlp中的相关代码（防止TMS系统使用）
* 保留mjs-parser
* 发现svmtagger的一个问题：2008年/m 12月/m 22日/m 应该是(nt）

2008-12-14
----------
* 使用mstparser替换原来马金山的parser
* 修改相应的xmlnlp及ltp...
* 修改nlp_style.xsl为v2.0

2008-09-23
----------
* TextClassify模块无论输入任何任何文本都输出“军事”。
* 删除模型中的SVMTestFile.dat即可（每次处理都会写这个文件）。
* 查看源代码发现：PatternChangeForSVM::OutputSVMDataFile()打开这个文件，没有判断是否打开。而这个文件被我无意中变成了只读文件。修改这个函数，判断文件是否成功打开，并且输出错误信息。

2008-07-07
----------
发布v1.5.0 

2008-07-05
----------
* srl处理某些文本的时候ARG的位置总是第1个。胡禹轩发现是模型问题，并且修正了这个错误。

2008-07-01
----------
* svmtagger简单处理英文串?/ws；ppmmm/ws；之前识别为nh。还需要改进：?/wp

2008-06-26
----------
* 增加了基于svmtool的汉语词性标注模块。
    * 接口：保留原来的int IRLAS()，完成词法分析的功能：基于图的分词+基于HMM的词性标注。增加：
        ```
        int SegmentWord();     # 分词，调用IRLAS中的分词
        int PosTag();    # 基于svmtool的词性标注。
        ```
* 保留IRLAS接口的原因：
    * 和以前版本兼容，用户如果不想使用新的模块，则不需要修改程序。
    * 方便用户对比两种词性标注方法。
* 用户如何修改原来的程序以使用新的词性标注模块？
    * 将用户程序中的IRLAS直接替换成PosTag即可。（根据模块的依赖关系，PosTag会自动调用SegmentWord。）
* 另外需要注意：如果调用了SegmentWord，那么只能通过PosTag进行词性标注。（IRLAS是一个集成的接口，要么做分词+词性标注，要么什么都不做。）

2008-04-23
----------
* 准备将srl更新为最新版本，出现问题，“摘要：”及“（”
* 发现可视化程序的一个小bug：“摘要：”

2008-04-24
----------
* 胡禹轩修改了srl的bug
* 可视化程序的bug没有修改，原因是按句子显示的时候没有EOS节点。暂时不考虑了。
* 升级LTP至1.4.3

2008-02-22
----------
* 付瑞吉修改NE模块中的一个bug，越界等

2008-01-29
----------
* 将CR更新为我本科时做的基于规则的方法。

2008-01-24
----------
* 修改了各个xxx_DLL.h中的宏定义

2008-01-21
----------
* 根据公司合同，修改了各个底层模块的接口（除CR外）。

2008-01-11
----------
* Parser中Parser.cpp中
```
float CParser::Smoothen(float ftd, float ftt, float ftttt, float ftw, float fwt, float fww)
```
中增加：`if (fProb <= 0.00000001) fProb = 0.00000001;`否则存在log(0)隐患。


2007-12-28
----------
* Parser_dll.h中原来的宏定义有问题。修改为：
```
#ifdef _WIN32     
    #undef PARSER_DLL_API
    #ifdef PARSER_DLL_API_EXPORT
        #define PARSER_DLL_API extern "C" _declspec(dllexport) 
    #else
        #define PARSER_DLL_API extern "C" _declspec(dllimport)
    #endif
#endif
```
* Parser增加接口Parse_with_postag()，这个接口直接提供词性细分类，parse预处理时再词性细分类了。

2007-12-08
----------
* 王丽杰修改xsl显示，使ARG等显示居中。
* 胡禹轩修改DepSRL.h DepSRL.cpp
* Arg的名字过长也影响美观，请禹轩输出的时候修改一下吧，ARG0->A0，ARGM-ADV->AM-ADV等等。

2007-12-06
----------
* 王丽杰修改xsl，显示srl的语义框架，采用绝对路径。这样给用户共享的xsl通过从服务器download文件，显示语义框架。
* 王丽杰修改了关于srl显示的bug。

2007-12-03
----------
* 胡禹轩修改了srl中overlapped的bug
* 上午修改IRLAS_DLL_x.cpp的时候使用到了MyLib.cpp。但是ltp中有很多MyLib.cpp，如：

  * _irlas/MyLib.cpp
  * __util/MyLib.cpp
  * _parser/MyLib.cpp

等，现在还没有统一。
* 我简单的将_irlas/MyLib.cpp替换__util/MyLib.cpp，导致出现了新的bug。因为_irlas/MyLib.cpp和__util/MyLib.cpp中convert_to_pair的实现不相同。完成功能是：`、/wp => [、][wp]`，_irlas/MyLib.cpp中的实现是错误的。
* 修订了这个bug。更新一下v1.4.1

2007-12-03
----------
* 王丽杰修改xsl文件，完善句法分析按句子显示时不能自动break的问题。
* xsl文件有一处需要改动一下，在“按句子显示”的“句法分析”下面定义的table，

```
body +='<table style="border: SOLID 0px black;word-break:break-all;" bgcolor="black" width="'+ widthTable +'" cellpadding="0" cellspacing="1">';
```

红色为新加入的，否则对于英文，table不会自动换行的，使得句法分析树有些错乱。
* 例句：参数errcode是来自函数regcomp()或regexec()的错误代码，而参数preg则是由函数regcomp()得到的编译结果，其目的是把格式化消息所必须的上下文提供给regerror()函数。在执行函数regerror()时，将按照参数errbuf_size指明的最大字节数，在errbuf缓冲区中填入格式化后的错误信息，同时返回错误信息的长度。
* NER的Bug：IRLAS的结果中可能会出现类似：

```
也/d 是/v 国内/nl SVM/ws 最好/d 的/u 学者/n 之/u 
一  /m 4/m 、/wp 数据/n 挖掘/v 中/nd 的/u 新/a 方法/n ：/wp 
```

`一  /m`，`：/wp`"/"前有空格。这样NE在内部处理的时候没有考虑到这个情况。也可以认为是IRLAS的bug。
* 修改了IRLAS_DLL_x.cpp文件，将分词结果中每个词中包含的' '去掉。
* 修改NER_DLL_x.cpp文件，修改了从`也/d#O 是/v#O 国内/nl#O SVM/ws#O 最好/d#O`抽取结果的方式。
* NEReg的返回值由void变为int，相应LTP.cpp中调用NEReg函数时也有所变化。


2007-12-02
----------
* LTP升级为v1.4.0包括网站程序

2007-12-01
----------
* NER模块更新，使用最大熵模型（付瑞吉）
* SRL模块更新使用最大熵模型（胡禹轩）
* SRL：(Release)
    * 加载数据资源数据，约15秒
    * 处理13K文件需要25秒
* SRL还有一个bug，ARG范围(beg end)重叠overlapped，让王丽杰修改xsl文件，绕过这个bug。

2007-11-23
----------
* 王丽杰 nlp_style.xsl中`action = "Try.py"`改为`action = "ltp_v1.3_start.py"`

2007-11-23
----------
* Parser MyLib.cpp中：
```
string itos(int i);
    char buf[4];
```
太小了。当i >= 1000时，便会发生异常。改为：`char buf[256];`

* 同时我对Parser中`void GetParseResults(vector < string >& vecWord, vector < string >& vecPOS, char * strOutput)
的代码进行了优化。

2007-11-22
----------
* NER NERtestDll_x.cpp中：
```
    char* presult = new char[5000];
```

当句子过长的时候，会出现内存越界。修改为：

```
    int nChar = 0;
    for (int i=0; i<(int)vecWord.size(); ++i)    {
        nChar += vecWord[i].size();
    }
    const int SZ = nChar + vecWord.size() * 10;
    char* presult = new char[SZ];
```

2007-11-22
----------
* Parser中，由于分句模块分出的句子太长，考虑将分句模块的句长减小，由
```
#define POLARIS_SENTENCE_LENGTH 0xFFFF
```
变为：
```
#define POLARIS_SENTENCE_LENGTH 0x1024
```
这样一句话最多约为500个汉字。和原来一样。

2007-11-22
----------
SDS中：`void SDS_TS::SelectSnt()` 定义：

```
    unsigned sntNum;
```
但是后面用到：
```
    sntNum = m_vctSntPairs_Score[summarySntNum].m_nSntNum - 1;
    if(sntNum >= 0) {
        ...
    }
```
此时当`m_vctSntPairs_Score[summarySntNum].m_nSntNum == 0`时：`sntNum = 0xFFFF;`

2007-11-22
----------
* 郭宇航找到一个bug：`藩`的第二个字节是 '['，Parser中：Extract()字符串处理时，没有考虑这个问题。
* 修改后的dll已经在网站上更新。

2007-11-22
----------
* parser_dll_x.cpp中`void Parse(vector < string >& vecWord, vector < string >& vecPOS, vector < pair<int,string> >& vecParse)`中原来为：
```
    char * csOutput = new char[vecWord.size() * 50];
```
现在改为：
```
    int i = 0;
    for (; i < vecWord.size(); ++i) {
        nChar += strlen(vecWord[i].c_str());
    }
    char * csOutput = new char[nChar * 2 + vecWord.size() * 32];
```

因为有的时候会输入`"------------------------------------"`或者很长的数字串，这样会造成内存越界问题。

2007-11-21
----------
v1.3.4
* 王丽杰对xsl文件进行更新

2007-08-31
----------
* 浙大仇光提出问题：如果文本中出现只包含空白符的行时，分词模块就会出错。修改`XML4NLP::CreateDOM`时的做法，在BuildParagraph之前，将每一行的句首及句尾的空白符全部去掉。

2007-07-16
----------
* 宋巍提出问题：修改分句模块：SplitSentence.cpp，原来认为如果一段话的分句结果中包含0个句子时，就表示错误。修改之后程序更加健壮了，即使遇到只含有空格的段落，也可以正确处理。

2007-06-27
----------
* 王健楠提出：xmlnlp.cpp中分段处理`CreateDOMFromString("中国\r\n\r\n美国")`，无法正常分段，于是将\r替换成\n

2007-06-23
----------
* 修改了模块、资源、网页等，统一正名为HIT IR Lab或者HIT-IR，更新了论坛地址，ltp的demo地址

2007-06-27
----------
* 郎君提出：修改nlp_style.xsl文件中网页顶部显示部分，更新了论坛地址，去掉了密码iloveirlab

2007-06-14
----------
* 分句对句子长度限制为1024，潘越群发现太小了
* 改成`#define POLARIS_SENTENCE_LENGTH 0xFFFF`

2007-04-30
----------
* 刘老师要求修改句法可视化的弧的指向，从head节点指向依存节点（和以前相反），因此使用2007-1-21为马金山师兄提供的xsl版本。
* 发布v1.3.1
* 发现还存在一些问题：按句子显示时，句法显示和上面的显示有重叠现象，原因是

2007-04-13
----------
* 发布v1.3
    1. IRLAS的extend_dict解密,不采用加密文件
    2. Parser越界错误
    3. 采用LTMLv2.0格式
    4. 发布python包
    5. 补充英文文档

* 由于LTMLv2.0中去掉了wsdexp属性，所以ltp_dll的接口需要改变一下。可以考虑提供两个版本：一个是没有去掉wsdexp的，另一个是去掉的。对于C接口（for python, perl...），需要wsdexp在DOM中存在，因此比较麻烦。所以v1.3中仍然采用LTMLv1.0，即旧的格式。
* 将网站上LTP更新为LTPv1.3。

2007-04-10
----------
* IRLAS的extend_dict解密，相应的加载过程也修改了Dictionary.cpp。(by 付瑞吉)

2007-04-04
----------
* Parser越界错误 Phrase.cpp

2007-01-21
----------
* 修改了nlp_style.xsl中parser的显示部分：
    1. 箭头从parent指向children
    2. 句法分析结果中，某一个词的句法角色未知，此时其parent为自己的word idx，不显示这个词对应的弧

2007-01-16
----------
* 将工程中所有的`GetMentionOfEnity`修改为`GetMentionOfEntity()`
* 在Parser.cpp中，函数void CParser::CreateLeaf(int i)使用了vector::erase(iter+0)操作，然后没有对iter进行赋值就直接继续使用。这是一个潜在的bug，在VC2005中体现出来。VC2005代码的安全性更高了。但是有一个问题：VC2005生成的Release版DLL只能用在Release工程中，Debug也是一样，无法在一个Debug工程中使用Release版的DLL。还需要进一步验证。实验室只有刘怀军师兄真正使用VC2005调用LTP。

2007-01-14
----------
* 增加了ltp_dll_for_python.dll，完成了Python Interface

2007-01-11
----------
* 对外正式发布LTPv1.2


2006-12-30
----------

* 通过反复验证：发现使用VC7生成DLL，如果接口含有string，如
```
_declspec (dllexport) void processString(const string &str)
{
    cout << str.size() << endl;
}
```
如果使用VC6程序调用，则会输出很大的数，可以称为乱码。怀疑在vc6和vc7.1在这方面不兼容，本身对string的实现就不一样。而如果用同一种平台，不会出错。此时如果使用string的复制等肯定会出错！为此修改几个LTP接口：（不能有string参数）。将__ltp_dll.h中的`LTP_DLL_API int CreateDOMFromString(const string &str);`修改成：
```
inline int CreateDOMFromString(const string &str)    // Due to incompatible between VC6 and VC7.1 in DLL
{
    return CreateDOMFromString(str.c_str());
}
```

改变了几个接口，使接口形式一致：
```
LTP_DLL_API int _GetPredArgToWord(vector<const char *> &vecType, 
                vector< pair<int, int> > &vecBegEnd,
                int paragraphIdx, 
                int sentenceIdx,
                int wordIdx);

LTP_DLL_API int _GetPredArgToWord(vector<const char *> &vecType, 
                vector< pair<int, int> > &vecBegEnd,
                int globalSentIdx, int wordIdx);

LTP_DLL_API int _GetPredArgToWord(vector<const char *> &vecType, 
                vector< pair<int, int> > &vecBegEnd,
                int globalWordIdx);

int GetPredArgToWord(vector<const char *> &vecType,
                vector< pair<int, int> > &vecBegEnd, 
                int paragraphIdx,
                int sentenceIdx,
                int wordIdx);

int GetPredArgToWord(vector<const char *> &vecType, 
                vector< pair<int, int> > &vecBegEnd,
                int globalSentIdx,
                int wordIdx);

int GetPredArgToWord(vector<const char *> &vecType,
                vector< pair<int, int> > &vecBegEnd,
                int globalWordIdx);
```

2006-12-30
----------
* 遇到了问题：ltp_dll调用sds时，使用了system("del tmp.xml")，却没有删除！可能是文件夹只读的原因（无法改回去）。
* 分词问题（导致NE问题，进而cr也会产生一些问题，现在cr只是随便应付一下，不至于无法运行）：

```
                <word id="10" cont="展示" pos="v" parent="9" relate="VOB" ne="O" />
                <word id="11" cont="的" pos="u" parent="15" relate="ATT" ne="O" />
                <word id="12" cont="穆" pos="j" parent="13" relate="ATT" ne="B-Nh" />
                <word id="13" cont="哈" pos="j" parent="14" relate="ATT" ne="B-Nh" />
                <word id="14" cont="吉尔" pos="nh" parent="15" relate="ATT" ne="B-Nh" />
                <word id="15" cont="照片" pos="n" parent="16" relate="DE" ne="O" />
                <word id="16" cont="的" pos="u" parent="17" relate="ATT" ne="O" />
                <word id="17" cont="相貌" pos="n" parent="18" relate="SBV" ne="O" />
                
                </word>
                <word id="10" cont="马斯理" pos="nh" parent="14" relate="ATT" ne="S-Nh" />
                <word id="11" cont="过去" pos="nt" parent="12" relate="ADV" ne="O" />
                <word id="12" cont="经历" pos="v" parent="13" relate="DE" ne="O">
                    <arg id="0" type="ArgM-TMP" beg="11" end="11" />
                </word>
                <word id="13" cont="的" pos="u" parent="14" relate="ATT" ne="O" />
                <word id="14" cont="细节" pos="n" parent="9" relate="VOB" ne="O" />
                <word id="15" cont="，" pos="wp" parent="-2" relate="PUN" ne="O" />
                <word id="16" cont="却" pos="d" parent="17" relate="ADV" ne="O" />
                <word id="17" cont="连" pos="v" parent="7" relate="VV" ne="O">
                    <arg id="0" type="Arg1" beg="18" end="20" />
                </word>
                <word id="18" cont="他" pos="r" parent="19" relate="DE" ne="O" />
                <word id="19" cont="的" pos="u" parent="20" relate="ATT" ne="O" />
                <word id="20" cont="真名" pos="n" parent="17" relate="VOB" ne="O" />
                <word id="21" cont="（" pos="wp" parent="-2" relate="PUN" ne="O" />
                <word id="22" cont="穆哈吉尔" pos="nh" parent="26" relate="IS" ne="S-Nh" />
                <word id="23" cont="）" pos="wp" parent="-2" relate="PUN" ne="O" />
                <word id="24" cont="都" pos="d" parent="26" relate="ADV" ne="O" />
                <word id="25" cont="不" pos="d" parent="26" relate="ADV" ne="O" />
                <word id="26" cont="知道" pos="v" parent="7" relate="VV" ne="O" />
                <word id="27" cont="。" pos="wp" parent="-2" relate="PUN" ne="O" />
```

* 增加了一层包装，不需要用户自己resize()。
* clear()不是basic_string的成员！因此XML4NLP类中的一些就需要修改一下。
    * 教训是：应该至少在VC6下也测试一遍。
    * 修改成`strParagraph = "";`
    * 另外：`if (0 != CheckRange(paragraphIdx)) -1;`居然可以编译通过！ 赶紧修改过来。

* VC6不支持语法：
```
class LTP
{
    static const unsigned int DO_XML = 1;
    static const unsigned int DO_SPLITSENTENCE = 1 << 1;
    static const unsigned int DO_IRLAS = 1 << 2;
}
```
修改成：
```
class LTP
{
    static const unsigned int DO_XML = 1;
    static const unsigned int DO_SPLITSENTENCE = 1 << 1;
}
```
然后在ltp.cpp中定义：
```
const unsigned int LTP::DO_XML = 1;
const unsigned int LTP::DO_SPLITSENTENCE = 1 << 1;
```

* 对inline函数的定义修改成规范的定义方式，在头文件中定义。
* 增加LTP_DLL接口：`LTP_DLL_API int CreateDOMFromString(const char *str);`
    * 如果没有这个接口，在VC6中调用`CreateDOMFromString("我是一个中国人"）`就会抛出异常。可能还是分配内存的问题。
  现在，VC7.1中生成的DLL就可以在VC6中使用了。

* 增加了接口：`LTP_DLL_API const char *GetParagraph(int paragraphIdx);`
* 修改：
```
void XML4NLP::ClearDOM()
{
    m_tiXmlDoc.Clear();

    m_document_t.documentPtr = NULL;
    m_document_t.vecParagraph_t.clear();
    m_note.nodePtr = NULL;
    m_summary.nodePtr = NULL;
    m_textclass.nodePtr = NULL;
    m_coref.nodePtr = NULL;
    m_coref.vecEntity.clear();        // 增加！

    m_vecBegWordIdxOfStns.clear();
    m_vecBegStnsIdxOfPara.clear();
}
```

2006-12-25
----------
* 由于林建国给的TextClassify新版本在VC7下面无法正常编译，因此我自行改了一下原来的版本，使得数据文件位置可配置。
* Parser我也改了一下。所有的数据文件路径都可配置。
* 修改了LTP配置文件的内容。

2006-12-21
----------
* 增加了所有需要的接口以支持用户对DOM的get操作。并且进行了初步的测试。现在系统已经可以很方便的为实验室内部或外部使用。
* 祝慧佳修改了NERtest.cpp中`void NERtest::getNEstring(unsigned int& begpos, string& strOut)`函数。解决了
```
"田壮/nh 壮憨然/nh 一/m 笑/v ";
"炮轰/v 布什/nh 越战/j 中/j 退缩/v ";
```
有时无法正常显示的问题。
* 改变了WSD的部分接口，使得数据路径可配置。

2006-12-18
----------
1.增加接口。
    由于DLL分配内存时在自己的local heap上分配，导致DLL中分配的内存无法在DLL外部释放。因此类似`LTP_DLL_API int GetWordsFromSentence(vector<string> &vecWord, int sentenceIdx);`的接口都无法正常工作。
    添加类似`LTP_DLL_API int GetWordsFromSentence(vector<const char *> &vecWord, int sentenceIdx);`这样的接口。需要在传入vecWord前根据CountWordInSentence(sentenceIdx)的值resize vecWord的大小。

2006-12-16
----------
* 修改接口 `string XML4NLP::GetParagraph(int paragraphIdx) const;`为`int XML4NLP::GetParagraph(int paragraphIdx, string &strParagraph) const;`

2006-12-05
----------
* 修改xsl文件，以适应新的xml数据格式。(<para id=" ">...</para>)考虑是否采用<note>

2006-12-04
----------
* 准备发布v1.2。
* 支持以前的接口，提供更多的，更灵活的接口。
* 需要在LTPv_1_2_share中做一些必要的变化。

2006-10-24
----------
1. 为了对旧的LTML文件格式进行修改，增加了void CheckNoteForOldLtml();完成对旧的LTML检测哪些模块已经被调用了，然后相应的添加note标记。
2. 对SetInfoToSentence中vecInfo大小和word num不相等时的错误提示进行了修改。
3. 实现了依赖关系
4. 增加了irlas,ne,和sds的选项设置功能，但是由于还有一些设计上和模块自身的问题，推荐不要使用，而使用默认的选项设置。sds的选项设置还提供了一个重载函数实现。
5. 增加了错误判断，代码更加安全

2006-10-18
----------
* 初步实现LTP的框架。

* 为了防止内存泄漏，修改XML4NLP这几个函数：
    * 增加：`~XML4NLP() {Clear();}`
    * `BuildDOMFrame()`
    * 三个`LoadDOMFrom...()`
* 其中BuildDOMFrame，和其中的两个LoadDOMFrom必须改，否则会有内存泄漏。

2006-10-22
----------
* 由于XML4NLP的结构存在根本上的问题：
    * 采用继承的方式使用TinyXml。
    * 在程序中出现了大量的static_cast<InheritedClass *>(BaseClassObjPtr)的现象；增加了程序复杂度；
    * 无法方便的换解析器（如果要换的话）；
    * 抗错误操作的能力不强，异常机制也没有。

* 鉴于以上原因，重写XML4NLP，但是接口保持不变。
* 主要的改变：
    * 不采用继承机制，而将TinyXml作为一个内部成员使用；
    * 像以前一样采用树状结构，将DOM上对应doc, para, sent, word的节点存储在vector中；
    * 增加了note及其操作；
    * 改变以前的做法：如果还未分句，则将段落内容作为第一句。现在采用如果没有分段，则没有分句节点。
    * 大量的错误判断，但是还是没有使用异常；
    * 将所有的标记都以const char * const在头文件中声明，以后改变标记比较容易；

* 存在问题：
    1. `MapGlobalSentIdx2paraIdx_sentIdx()`, `MapGloablWordIdx2paraIdx_sentIdx_wordIdx()`还没有进行优化。
    2. inline的使用？为什么不能使用在头文件声明，cpp定义的方式？两个都使用了inline修饰符

2006-10-16
----------
* 生成VC6下的DLL，共享给用户。

2006-10-12
----------
* 解决NE和SDS(单文档自动文摘)两个模块不能重复调用的问题，原因是这两个模块的资源加载和释放存在问题。现在如下调用main2()就很顺利了。

```
void main()
{
    clock_t start = clock();
    cout << main2("S01.txt","S01.xml","ltpconfig.ini") << endl;
    cout << ((float)clock() - start) / CLOCKS_PER_SEC << endl;

    start = clock();
    cout << main2("S01.txt","S02.xml","ltpconfig.ini") << endl;
    cout << ((float)clock() - start) / CLOCKS_PER_SEC << endl;
}
```

我们会在下一次更新中，提供对各个模块更加灵活的调用方式。


之前输出参数在后，输入参数在前。

