[![LTP](https://img.shields.io/pypi/v/ltp?label=LTP4%20ALPHA)](https://pypi.org/project/ltp/)
![VERSION](https://img.shields.io/pypi/pyversions/ltp)
![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)
[![Documentation Status](https://readthedocs.org/projects/ltp/badge/?version=latest)](https://ltp.readthedocs.io/zh_CN/latest/?badge=latest)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ltp)](https://pypi.python.org/pypi/ltp)

# LTP 4 

LTP（Language Technology Platform） 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。

If you use any source codes included in this toolkit in your work, please kindly cite the following paper. The bibtex are listed below:
<pre>
@article{che2020n,
  title={N-LTP: A Open-source Neural Chinese Language Technology Platform with Pretrained Models},
  author={Che, Wanxiang and Feng, Yunlong and Qin, Libo and Liu, Ting},
  journal={arXiv preprint arXiv:2009.11616},
  year={2020}
}
</pre>

## 快速使用

```python
from ltp import LTP
ltp = LTP() # 默认加载 Small 模型
seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
pos = ltp.pos(hidden)
ner = ltp.ner(hidden)
srl = ltp.srl(hidden)
dep = ltp.dep(hidden)
sdp = ltp.sdp(hidden)
```

**[详细说明](docs/quickstart.rst)**

## 指标

|      模型       | 分词  | 词性  | 命名实体 | 语义角色 | 依存句法 | 语义依存 | 速度(句/S) |
| :-------------: | :---: | :---: | :------: | :------: | :------: | :------: | :--------: |
| LTP 4.0 (Base)  | 98.7  | 98.5  |   95.4   |   80.6   |   89.5   |   75.2   |            |
| LTP 4.0 (Small) | 98.4  | 98.2  |   94.3   |   78.4   |   88.3   |   74.7   |   12.58    |
| LTP 4.0 (Tiny)  | 96.8  | 97.1  |   91.6   |   70.9   |   83.8   |   70.1   |   29.53    |

**[模型下载地址](MODELS.md)**

## 模型算法

+ 分词: Electra Small<sup>[1](#RELTRANS)</sup> + Linear
+ 词性: Electra Small + Linear
+ 命名实体: Electra Small + Relative Transformer<sup>[2](#RELTRANS)</sup> + Linear
+ 依存句法: Electra Small + BiAffine + Eisner<sup>[3](#Eisner)</sup>
+ 语义依存: Electra Small + BiAffine
+ 语义角色: Electra Small + BiAffine + CRF

## 构建 Wheel 包

```shell script
python setup.py sdist bdist_wheel
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
