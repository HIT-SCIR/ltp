使用训练套件
============

分词训练套件otcws用法
-----------------------

otcws是ltp分词模型的训练套件，用户可以使用otcws训练获得ltp的分词模型。otcws支持从人工切分数据中训练分词模型和调用分词模型对句子进行切分。人工切分的句子的样例如下：::

	对外	，	他们	代表	国家	。

编译之后，在tools/train下面会产生名为otcws的二进制程序。运行可见::

    $ ./tools/train/otcws 
    otcws in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    Training and testing suite for Chinese word segmentation

    usage: ./otcws [learn|customized-learn|test|customized-test|dump] <options>

其中第二个参数调用训练(learn)或测试(test)或可视化模型(dump)，对于customized-learn以及customized-test，请参考 :ref:`customized-cws-reference-label`

训练一个模型
~~~~~~~~~~~~

如果进行模型训练(learn)，::

    $ ./tools/train/otcws learn
    otcws(learn) in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    Training suite for Chinese word segmentation
    
    usage: ./otcws learn <options>
    
    options:
      --model arg                  The prefix of the model file, model will be 
                                   stored as model.$iter.
      --reference arg              The path to the reference file.
      --development arg            The path to the development file.
      --algorithm arg              The learning algorithm
                                    - ap: averaged perceptron
                                    - pa: passive aggressive [default]
      --max-iter arg               The number of iteration [default=10].
      --rare-feature-threshold arg The threshold for rare feature, used in model 
                                   truncation. [default=0]
      --dump-details arg           Save the detailed model, used in incremental 
                                   training. [default=false]
      -h [ --help ]                Show help information

其中：

* reference：指定训练集文件
* development：指定开发集文件
* algorithm：指定参数学习方法，现在LTP在线学习框架支持两种参数学习方法，分别是passive aggressive(pa)和average perceptron(ap)。
* model：指定输出模型文件名前缀，模型采用model.$iter方式命名
* max-iter：指定最大迭代次数
* rare-feature-threshold：模型裁剪力度，如果rare-feature-threshold为0，则只去掉为0的特征；rare-feature-threshold；如果大于0时将进一步去掉更新次数低于阈值的特征。关于模型裁剪算法细节，请参考 :ref:`truncate-reference-label` 部分。
* dump-details：指定保存模型时输出所有模型信息，这一参数用于 :ref:`customized-cws-reference-label` ，具体请参考 :ref:`customized-cws-reference-label` 。

需要注意的是，reference和development都需要是人工切分的句子。

测试模型/切分句子
~~~~~~~~~~~~~~~~~

如果进行模型测试(test)，::

    $ ./tools/train/otcws test
    otcws(test) in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    Testing suite for Chinese word segmentation
    
    usage: ./otcws test <options>
    
    options:
      --model arg           The path to the model file.
      --lexicon arg         The lexicon file, (optional, if configured, constrained
                            decoding will be performed).
      --input arg           The path to the reference file.
      --evaluate arg        if configured, perform evaluation, input words in 
                            sentence should be separated by space.
      -h [ --help ]         Show help information

其中，

* model：指定模型文件位置
* lexicon：指定外部词典位置，这个参数是可选的
* input：指定输入文件
* evaluate：true或false，指定是否进行切词结果准确率的评价。如果进行评价，输入的文件应该是人工切分的数据。

外部词典 (lexicon) 格式请参考 :ref:`ltpexlex-reference-label` 。

切分结果将输入到标准io中。

词性标注训练套件otpos用法
--------------------------

otpos是ltp分词模型的训练套件，用户可以使用otpos训练获得ltp的分词模型。otpos支持从人工切分并标注词性的数据中训练词性标注模型和调用词性标注模型对切分好的句子进行词性标注。人工标注的词性标注句子样例如下：::

	对外_v	，_wp	他们_r	代表_v	国家_n	。_wp

编译之后，在tools/train下面会产生名为otpos的二进制程序。otpos的使用方法与otcws非常相似，同名参数含义也完全相同。其中不同之处在于词性标注模块的外部词典含义与分词的外部词典含义不同。::

    $ ./tools/train/otpos test
    otpos(test) in LTP 3.3.2 - (C) 2012-2016 HIT-SCIR
    Testing suite for Part of Speech Tagging

    usage: ./otpos test <options>

    options::
      --model arg           The path to the model file.
      --lexicon arg         The lexicon file, (optional, if configured, constrained
                            decoding will be performed).
      --input arg           The path to the reference file.
      --evaluate arg        if configured, perform evaluation, input should contain
                            '_' concatenated tag
      -h [ --help ]         Show help information

外部词典 (lexicon) 格式请参考 :ref:`ltpexlex-reference-label` 。

命名实体识别训练套件otner用法
-------------------------------

otner是ltp命名实体识别模型的训练套件，用户可以使用otner训练获得ltp的命名实体识别模型。otner支持从人工标注的数据中训练命名实体识别模型和调用命名实体识别模型对句子进行标注。人工标注的句子的样例如下：::

	党中央/ni#B-Ni 国务院/ni#E-Ni 要求/v#O ，/wp#O 动员/v#O 全党/n#O 和/c#O 全/a#O社会/n#O 的/u#O 力量/n#O

编译之后，在tools/train下面会产生名为otner的二进制程序。otner的使用方法与otcws非常相似，同名参数含义也完全相同。

依存句法分析训练套件nndepparser用法
-----------------------------------

nndepparser是ltp神经网络依存句法分析模型的训练套件，用户可以使用nndepparser训练获得ltp的依存句法分析模型。nndepparser分别支持从人工标注依存句法的数据中训练依存句法分析模型和调用依存句法分析模型对句子进行依存句法分析。人工标注的词性标注依存句法的句子遵从conll格式，其样例如下：::

	1       对外    _       v       _       _       4       ADV     _       _
	2       ，      _       wp      _       _       1       WP      _       _
	3       他们    _       r       _       _       4       SBV     _       _
	4       代表    _       v       _       _       0       HED     _       _
	5       国家    _       n       _       _       4       VOB     _       _
	6       。      _       wp      _       _       4       WP      _       _

编译之后，在tools/train下面会产生名为nndepparser的二进制程序。调用方法是::

	./nndepparser [learn|test] <options>

训练一个parser
~~~~~~~~~~~~~~

运行./nndepparser learn，可见如下参数::

    $ ./tools/train/nndepparser learn
    nndepparser(learn) in ltp 3.3.2 - (c) 2012-2016 hit-scir
    training suite for neural network parser
    usage: ./nndepparser learn <options>

    options:
      --model arg               the path to the model.
      --embedding arg           the path to the embedding file.
      --reference arg           the path to the reference file.
      --development arg         the path to the development file.

      --init-range arg          the initialization range. [default=0.01]
      --word-cutoff arg         the frequency of rare word. [default=1]
      --max-iter arg            the number of max iteration. [default=20000]
      --batch-size arg          the size of batch. [default=10000]
      --hidden-size arg         the size of hidden layer. [default=200]
      --embedding-size arg      the size of embedding. [default=50]
      --features-number arg     the number of features. [default=48]
      --precomputed-number arg  the number of precomputed. [default=100000]
      --evaluation-stops arg    evaluation on per-iteration. [default=100]
      --ada-eps arg             the eps in adagrad. [defautl=1e-6]
      --ada-alpha arg           the alpha in adagrad. [default=0.01]
      --lambda arg              the regularizer parameter. [default=1e-8]
      --dropout-probability arg the probability for dropout. [default=0.5]
      --oracle arg              the oracle type
                                 - static: the static oracle [default]
                                 - nondet: the non-deterministic oracle
                                 - explore: the explore oracle.
      --save-intermediate arg   save the intermediate. [default=true]
      --fix-embeddings arg      fix the embeddings. [default=false]
      --use-distance arg        specify to use distance feature. [default=false]
      --use-valency arg         specify to use valency feature. [default=false]
      --use-cluster arg         specify to use cluster feature. [default=false]
      --cluster arg             specify the path to the cluster file.
      --root arg                the root tag. [default=root]
      --verbose                 logging more details.
      -h [ --help ]             show help information.

nndepparser具有较多参数，但大部分与Chen and Manning (2014)中的定义一致。希望使用nndepparser的用户需要首先阅读其论文。
另，经验表明，大部分参数采用默认值亦可取得较好的效果。
所以在不能明确各参数含义的情况下，可以直接使用默认值。

相较Chen and Manning (2014)，nndepparser中特有的参数包括：

* oracle：指定oracle函数类型，可选的oracle包括static，nondet和explore。一般来讲，explore效果最好，具体算法请参考Yoav et. al, (2014)
* use-distance：指定使用距离特征，具体参考Zhang and Nivre (2011)
* use-valency：指定使用valency特征，具体参考Zhang and Nivre (2011)
* use-cluster：指定使用词聚类特征，具体参考Guo et. al, (2015)
* root：根节点的deprel的类型，需要注意的是，当前版本nndepparser只能处理projective single-root的依存树。

参考文献
--------
- Danqi Chen and Christopher Manning, 2014, A Fast and Accurate Dependency Parser using Neural Networks, In Proc. of *EMNLP2014*
- Yue Zhang and Joakim Nivre, 2011, Transition-based Dependency Parsing with Rich Non-local Features, In Proc. of *ACL2011*
- Yoav Goldberg, Francesco Sartorioand Giorgio Satta, 2014, A Tabular Method for Dynamic Oracles in Transition-Based Parsing, In *TACL2014*
- Jiang Guo, Wanxiang Che, David Yarowsky, Haifeng Wang and Ting Liu, 2015, Cross-lingual Dependency Parsing Based on Distributed Representations, (to apper) In Proc. of *ACL2015*

