[![LTP](https://img.shields.io/pypi/v/ltp?label=LTP4%20ALPHA)](https://pypi.org/project/ltp/)
![VERSION](https://img.shields.io/pypi/pyversions/ltp)
![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)
[![Documentation Status](https://readthedocs.org/projects/ltp/badge/?version=latest)](https://ltp.readthedocs.io/zh_CN/latest/?badge=latest)

# LTP 4 

LTP（Language Technology Platform） 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。

## 快速使用

```python
from ltp import LTP
ltp = LTP() # 默认加载 Small 模型
# ltp = LTP(path = "base|small|tiny")
# sent_list = ltp.sent_split(inputs, flag="all", limit=510)
segment, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
pos = ltp.pos(hidden)
ner = ltp.ner(hidden)
srl = ltp.srl(hidden)
dep = ltp.dep(hidden)
sdp = ltp.sdp(hidden)
```

## 模型

### 模型下载

|   模型    |                      大小                      |
| :-------: | :--------------------------------------------: |
| Base(v2)  | [UNKNOWN](http://39.96.43.154/ltp/v2/base.tgz) |
| Small(v2) | [170MB](http://39.96.43.154/ltp/v2/small.tgz)  |
| Tiny(v2)  |  [34MB](http://39.96.43.154/ltp/v2/tiny.tgz)   |
| Small(v1) | [170MB](http://39.96.43.154/ltp/v1/small.tgz)  |
| Tiny(v1)  |  [34MB](http://39.96.43.154/ltp/v1/tiny.tgz)   |

**备注**: Tiny模型使用electra前三层进行初始化, 4.0.3.post1 版本包含对v1模型的fix，新版本请使用v2模型

### V2 指标

|      模型       | 分词  | 词性  | 命名实体 | 语义角色 | 依存句法 | 语义依存 | 速度(句/S) |
| :-------------: | :---: | :---: | :------: | :------: | :------: | :------: | :--------: |
| LTP 4.0 (Base)  |       |       |          |          |          |          |            |
| LTP 4.0 (Small) | 98.4  | 98.2  |   94.3   |   78.4   |   88.3   |   74.7   |   12.58    |
| LTP 4.0 (Tiny)  | 96.8  | 97.1  |   91.6   |   70.9   |   83.8   |   70.1   |   29.53    |

**备注**: 本版本SDP采用CCS2020语义依存分析语料，其他语料同V1

### V1 指标

|      模型       | 分词  | 词性  | 命名实体 |      语义角色      | 依存句法 | 语义依存 | 速度(句/S) | 模型大小 |
| :-------------: | :---: | :---: | :------: | :----------------: | :------: | :------: | :--------: | :------: |
|     LTP 3.X     | 97.8  | 98.3  |   94.1   | ~~77.92(Gold Pi)~~ |   81.1   | ~~78.9~~ |    2.75    |  1940M   |
| LTP 4.0 (Small) | 98.4  | 98.2  |   94.3   |    77.2(端到端)    |   88.0   |   79.9   |   12.58    |   171M   |
| LTP 4.0 (Tiny)  | 96.8  | 97.2  |   91.6   |    68.1(端到端)    |   82.6   |   75.5   |   29.53    |   34M    |

测试环境如下：

+ Python 3.7
+ LTP 4.0 Batch Size = 1
+ Centos 3.10.0-1062.9.1.el7.x86_64
+ Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz

**备注**: 速度数据在人民日报命名实体测试数据上获得，速度计算方式均为所有任务顺序执行的结果。另外，语义角色标注与语义依存新旧版采用的语料不相同，因此无法直接比较（新版语义依存使用Semeval 2016语料，语义角色标注使用CTB语料）。

## 模型算法

+ 分词: Electra Small<sup>[1](#RELTRANS)</sup> + Linear
+ 词性: Electra Small + Linear
+ 命名实体: Electra Small + Relative Transformer<sup>[2](#RELTRANS)</sup> + Linear
+ 依存句法: Electra Small + BiAffine + Eisner<sup>[3](#Eisner)</sup>
+ 语义依存: Electra Small + BiAffine
+ 语义角色: Electra Small + BiAffine + CRF

## 构建 Wheel 包

```shell script
python setup.py sdist
python -m twine upload dist/*
```

## 作者信息

+ 冯云龙 <<[ylfeng@ir.hit.edu.cn](mailto:ylfeng@ir.hit.edu.cn)>>

## 开源协议
1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。
2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。
3. 凡涉及付费问题，请发邮件到 car@ir.hit.edu.cn 洽商。
4. 如果您在 LTP 基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”. 同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。


## 脚注

+ <a name="RELTRANS">1</a>:: [Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)
+ <a name="RELTRANS">2</a>:: [TENER: Adapting Transformer Encoder for Named Entity Recognition](https://arxiv.org/abs/1911.04474)
+ <a name="Eisner">3</a>:: [A PyTorch implementation of "Deep Biaffine Attention for Neural Dependency Parsing"](https://github.com/yzhangcs/parser)
