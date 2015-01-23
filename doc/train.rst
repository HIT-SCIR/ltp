使用训练套件
============

分词训练套件otcws用法
-----------------------

otcws是ltp分词模型的训练套件，用户可以使用otcws训练获得ltp的分词模型。

编译之后，在tools/train下面会产生名为otcws的二进制程序。调用方法是::

	./otcws [config_file]

otcws分别支持从人工切分数据中训练分词模型和调用分词模型对句子进行切分。人工切分的句子的样例如下::

	对外	，	他们	代表	国家	。

otcws主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示::

	[train]
	train-file = data/ctb5-train.seg
	holdout-file = data/ctb5-holdout.seg
	algorithm = pa
	model-name = model/ctb5-seg
	max-iter = 5
	rare-feature-threshold = 0

其中，

* [train] 配置组指定执行训练
	* train-file 配置项指定训练集文件
	* holdout-file 配置项指定开发集文件
	* algorithm 指定参数学习方法，现在otcws支持两种参数学习方法，分别是passive aggressive(pa)和average perceptron(ap)。
	* model-name 指定输出模型文件名
	* max-iter 指定最大迭代次数
	* rare-feature-threshold 配置裁剪力度，如果rare-feature-threshold为0，则只去掉为0的特征；rare-feature-threshold；如果大于0时将进一步去掉更新次数低于阈值的特征

测试配置的配置文件样例如下所示::

	[test]
	test-file = data/ctb5-test.seg
	model-file = model/ctb5-seg.4.model

其中，

* [test] 配置组指定执行测试
	* test-file 指定测试文件
	* model-file 指定模型文件位置


切分结果将输入到标准io中。

(\*[train]与[test]两个配置组不能同时存在)

词性标注训练套件otpos用法
--------------------------

otpos是ltp分词模型的训练套件，用户可以使用otpos训练获得ltp的分词模型。

编译之后，在tools/train下面会产生名为otpos的二进制程序。调用方法是::

	./otpos [config_file]

otpos分别支持从人工切分并标注词性的数据中训练词性标注模型和调用词性标注模型对切分好的句子进行词性标注。人工标注的词性标注句子样例如下::

	对外_v	，_wp	他们_r	代表_v	国家_n	。_wp

otpos主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示::

	[train]
	train-file = data/ctb5-train.pos
	holdout-file = data/ctb5-holdout.pos
	algorithm = pa
	model-name = model/ctb5-pos
	max-iter = 5

其中，

* [train] 配置组指定执行训练
	* train-file 配置项指定训练集文件
	* holdout-file 配置项指定开发集文件
	* algorithm 指定参数学习方法，现在otpos支持两种参数学习方法，分别是passive aggressive(pa)和average perceptron(ap)。
	* model-name 指定输出模型文件名
	* max-iter 指定最大迭代次数
	* rare-feature-threshold 配置裁剪力度，如果rare-feature-threshold为0，则只去掉为0的特征；rare-feature-threshold；如果大于0时将进一步去掉更新次数低于阈值的特征

测试配置的配置文件样例如下所示::

	[test]
	test-file = data/ctb5-test.pos
	model-file = model/ctb5-pos.3.model
	lexicon-file = lexicon/pos-lexicon.constrain

其中，

* [test] 配置组指定执行测试
	* test-file 指定测试文件
	* model-file 指定模型文件位置
	* lexicon-file 指定外部词典文件位置（此项可以不配置）

lexicon-file文件样例如下所示。每行指定一个词，第一列指定单词，第二列之后指定该词的候选词性（可以有多项，每一项占一列），列与列之间用空格区分。

	雷人 v a
	】 wp

词性标注结果将输入到标准io中。

(\*[train]与[test]两个配置组不能同时存在)

命名实体识别训练套件otner用法
-------------------------------

otner是ltp命名实体识别模型的训练套件，用户可以使用otner训练获得ltp的命名实体识别模型。

编译之后，在tools/train下面会产生名为otner的二进制程序。调用方法是::

	./otner [config_file]

otner分别支持从人工标注的数据中训练命名实体识别模型和调用命名实体识别模型对句子进行标注。人工标注的句子的样例如下::

	党中央/ni#B-Ni 国务院/ni#E-Ni 要求/v#O ，/wp#O 动员/v#O 全党/n#O 和/c#O 全/a#O社会/n#O 的/u#O 力量/n#O

Otner主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示::

	[train]
	train-file = data/ctb5-train.ner
	holdout-file = data/ctb5-holdout.ner
	algorithm = pa
	model-name = model/ctb5-ner
	max-iter = 5

其中，

* [train] 配置组指定执行训练
	* train-file 配置项指定训练集文件
	* holdout-file 配置项指定开发集文件
	* algorithm 指定参数学习方法，现在otner支持两种参数学习方法，分别是passive aggressive（pa）和average perceptron（ap）。
	* model-name 指定输出模型文件名
	* max-iter 指定最大迭代次数

测试配置的配置文件样例如下所示::

	[test]
	test-file = data/ctb5-test.ner
	model-file = model/ctb5-ner.4.model

其中，

* [test] 配置组指定执行测试
	* test-file 指定测试文件
	* model-file 指定模型文件位置

命名实体识别结果将输入到标准io中。

（\*[train]与[test]两个配置组不能同时存在）

依存句法分析训练套件lgdpj用法
------------------------------

lgdpj是ltp依存句法分析模型的训练套件，用户可以使用lgdpj训练获得ltp的依存句法分析模型。

编译之后，在tools/train下面会产生名为lgdpj的二进制程序。调用方法是::

	./lgdpj [config_file]

lgdpj分别支持从人工标注依存句法的数据中训练依存句法分析模型和调用依存句法分析模型对句子进行依存句法分析。人工标注的词性标注依存句法的句子遵从conll格式，其样例如下::

	1       对外    _       v       _       _       4       ADV     _       _
	2       ，      _       wp      _       _       1       WP      _       _
	3       他们    _       r       _       _       4       SBV     _       _
	4       代表    _       v       _       _       0       HED     _       _
	5       国家    _       n       _       _       4       VOB     _       _
	6       。      _       wp      _       _       4       WP      _       _

lgdpj主要通过配置文件指定执行的工作，其中主要有两类配置文件：训练配置和测试配置。

训练配置的配置文件样例如下所示::

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
	rare-feature-threshold = 0

其中，

* [mode] 配置组中
	* labeled 表示是否使用有label的依存句法分析
	* decoder-name 表示采用的解码算法，现在lgdpj支持三种解码算法，分别是1o，2o-sib，2o-carreras
* [feature] 配置组指定使用的特征
* [train] 配置组指定执行训练
	* train-file 配置项指定训练集文件
	* holdout-file 配置项指定开发集文件
	* algorithm 指定参数学习方法，现在lgdpj支持两种参数学习方法，分别是passive aggressive(pa)和average perceptron(ap)。
	* model-name 指定输出模型文件名
	* max-iter 指定最大迭代次数
	* rare-feature-threshold 配置裁剪力度，如果rare-feature-threshold为0，则只去掉为0的特征；rare-feature-threshold；如果大于0时将进一步去掉更新次数低于阈值的特征

测试配置的配置文件样例如下所示::

	[test]
	test-file = data/conll/ldc-test.conll
	model-file = model/parser/ldc-o2carreras.2.model

其中，

* [test] 配置组指定执行测试
	* test-file 指定测试文件
	* model-file 指定模型文件位置

依存句法分析结果将输入到标准io中。
