| Language                             | version                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Python](python/interface/README.md) | [![LTP](https://img.shields.io/pypi/v/ltp?label=LTP)](https://pypi.org/project/ltp) [![LTP-Core](https://img.shields.io/pypi/v/ltp-core?label=LTP-Core)](https://pypi.org/project/ltp-core)   [![LTP-Extension](https://img.shields.io/pypi/v/ltp-extension?label=LTP-Extension)](https://pypi.org/project/ltp-extension) |
| [Rust](rust/ltp/README.md)           | [![LTP](https://img.shields.io/crates/v/ltp?label=LTP)](https://crates.io/crates/ltp)                                                                                                                                                                                                                                     |

# LTP extension For Python

LTP for Rust 对 Python 的绑定，用于提升 LTP 的速度，以及加入传统机器学习算法实现的中文信息处理工具。

| method | ltp 3.0(c++) | ap(1) | ap(8) | pa    | pa-i(0.5) | pa-ii(0.5) |
| ------ | ------------ | ----- | ----- | ----- | --------- | ---------- |
| cws    | 97.83        | 97.93 | 97.67 | 97.90 | 97.90     | 97.93      |
| pos    | 98.35        | 98.41 | 98.30 | 98.39 | 98.39     | 98.38      |
| ner    | 94.17        | 94.28 | 93.42 | 94.02 | 94.06     | 93.95      |

## 自行编译安装

```bash
maturin build --release -m python/extension/Cargo.toml --out dist --no-default-features --features="malloc"
# or 针对cpu优化
maturin build --release -m python/extension/Cargo.toml --out dist --no-default-features --features="malloc" -- -C target-cpu=native
```

## features

- [x] 分句
- [x] 任务
  - [x] 中文分词(cws)
    - [ ] 对数字、英文、网址、邮件的处理
    - [x] 支持自定义词典
  - [x] 词性标注(pos)
    - [ ] 支持自定义词典
  - [x] 命名实体识别(ner)
- [x] 算法
  - [x] 平均感知机(ap)
    - [x] 单线程平均感知机
    - [x] 多线程平均感知机
  - [x] 被动攻击算法(pa)
- [ ] 模型量化
- [ ] 在线学习
- [ ] 增量学习

## 性能测试

### 评测环境

- Python 3.10
- MacBook Pro (16-inch, 2019)
- 处理器: 2.6 GHz 六核Intel Core i7
- 内存: 16 GB 2667 MHz DDR4

> 注: 速度测试文件大小为 33.85 MB / 305041 行

### 分词

我们选择Jieba、Pkuseg、Thulac等国内代表分词软件与 LTP 做性能比较，根据第二届国际汉语分词测评发布的国际中文分词测评标准，对不同软件进行了速度和准确率测试。

在第二届国际汉语分词测评中，共有四家单位提供的测试语料（Academia Sinica、 City University 、Peking University(PKU)
、Microsoft Research(MSR)）, 在评测提供的资源[icwb2-data](http://sighan.cs.uchicago.edu/bakeoff2005/)
中包含了来自这四家单位的训练集（icwb2-data/training）、测试集（icwb2-data/testing）,
以及根据各自分词标准而提供的相应测试集的标准答案（icwb2-data/gold）．在icwb2-data/scripts目录下含有对分词进行自动评分的perl脚本score。

我们在统一测试环境下，对若干流行分词软件和 LTP 进行了测试，使用的模型为各分词软件自带模型。在PKU和MSR测试集评测结果如下：

| Algorithm                                                                    | Speed(KB/s) |  PKU(F1) |  MSR(F1) |
| ---------------------------------------------------------------------------- | ----------: | -------: | -------: |
| [Jieba](https://github.com/fxsjy/jieba)                                      |      982.49 |     81.8 |     81.3 |
| [Pkuseg](https://github.com/lancopku/pkuseg-python)                          |      109.72 |     93.4 |     87.3 |
| [Thulac](https://github.com/thunlp/THULAC-Python)                            |       48.13 |     94.0 |     87.9 |
| [Thulac\[Fast\]](https://github.com/thunlp/THULAC-Python)                    |     1133.21 |       同上 |       同上 |
| [LTP 3(pyltp)](https://github.com/HIT-SCIR/pyltp)                            |      451.20 | **95.3** | **88.3** |
| [LTP legacy(1)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  | **1603.63** |     95.2 |     87.7 |
| [LTP legacy(2)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |     2869.42 |       同上 |       同上 |
| [LTP legacy(4)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |     4949.38 |       同上 |       同上 |
| [LTP legacy(8)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |     6803.88 |       同上 |       同上 |
| [LTP legacy(16)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension) | **7745.16** |       同上 |       同上 |

> **注：括号内为线程数量**

> **注2：Jieba的词表是在人民日报数据集上统计的**

### 流水线

除了分词以外，我们也测试了 LTP 三个任务（分词、词性标注、命名实体识别）流水线的速度：

| Algorithm                                                                    | Speed(KB/s) |
| ---------------------------------------------------------------------------- | ----------: |
| [LTP 3(pyltp)](https://github.com/HIT-SCIR/pyltp)                            |      153.10 |
| [LTP legacy(1)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |      508.74 |
| [LTP legacy(2)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |      899.25 |
| [LTP legacy(4)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |     1598.03 |
| [LTP legacy(8)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension)  |     2267.48 |
| [LTP legacy(16)](https://github.com/HIT-SCIR/ltp/tree/main/python/extension) |     2452.34 |

> **注：括号内为线程数量**

> **注2：速度数据在人民日报命名实体测试数据上获得，速度计算方式均为所有任务顺序执行的结果。**
