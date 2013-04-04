# LTP Training Wrapper

简介
----
这个脚本是LTP使用的训练模块的一种封装。封装的模块包括

 * 分词(Word Segmentation)
 * 词性标注(POS Tagging)
 * 依存句法分析(Dependency Parsing)
 * 语义角色标注(Semantic Role Labeling)

这个封装在linux下使用python 2.7开发，并未在其他python版本下进行测试(欢迎使用者为我们提供其他版本的测试结果)。

使用
----

### 编译结果

编译过程会自动在`tools/train`路径下产生

* `crf_learn`
* `maxent`
* `SRLExtract`
* `SRLGetInstance`
* `gparser`
* `svm_light`

6个可执行程序。

### 简要用法

这个封装提供如下一些支持：

* 构建模型：训练指定的模型并将模型暂存在build路径下

```
./ltp-model build [ws|pos|srl|dp] [OPTIONS]
```

* 部署模型：将训练的模型以及配置文件拷贝到`ltp_data`路径下

```
./ltp-model install
```

* 删除模型：删除训练出的模型

```
./ltp-model clean
```

### 配置文件

程序在运行过程中会自动产生一个名为`ltp-model.json`的临时文件，用以存储配置，请不要对其进行修改。

构建分词模型
------------

首先准备分好词的文本，词间使用空格断开。每句一行。例如：

```
上海 浦东 开发 与 法制 建设 同步
新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
...
```

然后运行脚本，训练分词模型。

```
./ltp-model build ws --train=<your-train-file>
```

这里需要注意的是，脚本默认输入文本以utf8编码。如果输入文本是其他编码，可以采用`--encoding=<encoding>`参数来指定编码。

分词模型使用[CRFPP](http://crfpp.googlecode.com/svn/trunk/doc/index.html)构建，这个封装同时也支持`crf_learn`中对应的参数。具体可以使用`crf_learn`命令检查


构建词性标注模型
----------------

首先准备分好词并进行词性标注的文本，词间使用空格断开，词与词性之间用`_`分割。每句一行。例如：

```
新华社_NR 北京_NR 九月_NT 一日_NT 电_NN （_PU 记者_NN 吴锦才_NR ）_PU
第三_OD 步_M 是_VC ７月份_NT 赴_VV 美国_NR 参加_VV 世界_NN 四_CD 强_NN 赛_NN 。_PU
...
```

然后运行脚本，训练词性标注模型。

```
./ltp-model build pos --train=<your-train-file>
```

构建依存句法分析模型
--------------------

首先准备ConllX格式的依存句法分语料，例如：

```
1       上海    _       NR      _       _       2       NMOD    _       _
2       浦东    _       NR      _       _       6       NMOD    _       _
3       开发    _       NN      _       _       6       NMOD    _       _
4       与      _       CC      _       _       6       NMOD    _       _
5       法制    _       NN      _       _       6       NMOD    _       _
6       建设    _       NN      _       _       7       SUB     _       _
7       同步    _       VV      _       _       0       ROOT    _       _
```

然后运行脚本，训练依存句法模型。

```
./ltp-model build dp --train=<your-train-file>
```

构建语义角色标注模型
--------------------

首先准备好语义角色标注的文本，文本依照ConllX格式定义，每词一行，句子间用空格断开。例如：

```
1  中国	中国	中国	ns	ns	_	_	3	3	ATT	ATT	_	_	_
2	进出口	进出口	进出口	v	v	_	_	3	3	ATT	ATT	_	_	_
3	银行	银行	银行	n	n	_	_	6	6	SBV	SBV	_	_	A0
4	与	与	与	p	p	_	_	6	6	ADV	ADV	_	_	_
5	中国银行	中国银行	中国银行	ni	ni	_	_	4	4	POB	POB	_	_	_
6	加强	加强	加强	v	v	_	_	0	0	HED	HED	Y	_	_
7	合作	合作	合作	v	v	_	_	6	6	VOB	VOB	_	_	A1

1	新华社	新华社	新华社	ni	ni	_	_	5	5	ATT	ATT	_	_
2	北京	北京	北京	ns	ns	_	_	5	5	ATT	ATT	_	_
3	十二月	十二月	十二月	nt	nt	_	_	4	4	ATT	ATT	_	_
...
```
然后运行脚本，训练语义角色标注模型。

```
./ltp-model build srl --train=<your-train-file>
```

