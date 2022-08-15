![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)

| Language | version                                                                                       |
|----------|-----------------------------------------------------------------------------------------------|
| Python   | [![LTP](https://img.shields.io/pypi/v/ltp?label=LTP4%20ALPHA)](https://pypi.org/project/ltp/) |
| Rust     |                                                                                               |

# LTP 4

LTP（Language Technology Platform） 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。

If you use any source codes included in this toolkit in your work, please kindly cite the following paper. The bibtex
are listed below:
<pre>
@article{che2020n,
  title={N-LTP: A Open-source Neural Chinese Language Technology Platform with Pretrained Models},
  author={Che, Wanxiang and Feng, Yunlong and Qin, Libo and Liu, Ting},
  journal={arXiv preprint arXiv:2009.11616},
  year={2020}
}
</pre>

**参考书：**
由哈工大社会计算与信息检索研究中心（HIT-SCIR）的多位学者共同编著的《[自然语言处理：基于预训练模型的方法](https://item.jd.com/13344628.html)
》（作者：车万翔、郭江、崔一鸣；主审：刘挺）一书现已正式出版，该书重点介绍了新的基于预训练模型的自然语言处理技术，包括基础知识、预训练词向量和预训练模型三大部分，可供广大LTP用户学习参考。

## 快速使用

### [Python](python/interface/README.md)

```python
from ltp import LTP

ltp = LTP()  # 默认加载 Small 模型
# ltp = LTP(pretrained_model_name_or_path="LTP/small")
# 另外也可以接受一些已注册可自动下载的模型名(https://huggingface.co/LTP): 
# 使用字典结果
output = ltp.pipeline(
    ["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp"]
)
print(output.cws)
print(output.pos)
print(output.sdp)

# 传统算法，比较快，但是精度略低
ltp = LTP("LTP/legacy")
cws, pos, ner = ltp.pipeline(
    ["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner"]
).to_tuple()
print(cws, pos, ner)
```

**[详细说明](python/interface/docs/quickstart.rst)**

### [Rust](rust/ltp/README.md)

## 指标

|    模型            | 分词    |  词性   |   命名实体   | 语义角色  | 依存句法  | 语义依存  |           速度(句/S)           |
|:----------------:|:-----:|:-----:|:--------:|:-----:|:-----:|:-----:|:---------------------------:|
|  LTP 4.0 (Base)  | 98.7  | 98.5  |   95.4   | 80.6  | 89.5  | 75.2  |            39.12            |
| LTP 4.0 (Base1)  | 99.22 | 98.73 |  96.39   | 79.28 | 89.57 | 76.57 |            --.--            |
| LTP 4.0 (Base2)  | 99.18 | 98.69 |  95.97   | 79.49 | 90.19 | 76.62 |            --.--            |
| LTP 4.0 (Small)  | 98.4  | 98.2  |   94.3   | 78.4  | 88.3  | 74.7  |            43.13            |
|  LTP 4.0 (Tiny)  | 96.8  | 97.1  |   91.6   | 70.9  | 83.8  | 70.1  |            53.22            |

|    模型            | 分词    |  词性   |   命名实体   | 速度(KB/s) |
|:----------------:|:-----:|:-----:|:--------:|:--------:|
| LTP 4.0 ([Legacy](rust/ltp/README.md)) | 97.93 | 98.41 |  94.28   |  1318.84 |

**[模型下载地址](https://huggingface.co/LTP)**

## 构建 Wheel 包

```shell script
make bdist
```

## 其他语言绑定

**目前仅支持传统学习算法**

+ [Rust](rust/ltp)
+ [C/C++](rust/ltp-cffi)

## 作者信息

+ 冯云龙 <<[ylfeng@ir.hit.edu.cn](mailto:ylfeng@ir.hit.edu.cn)>>

## 开源协议

1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。
2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。
3. 凡涉及付费问题，请发邮件到 car@ir.hit.edu.cn 洽商。
4. 如果您在 LTP 基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”.
   同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。
