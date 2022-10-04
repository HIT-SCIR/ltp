![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)

| Language                             | version                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Python](python/interface/README.md) | [![LTP](https://img.shields.io/pypi/v/ltp?label=LTP)](https://pypi.org/project/ltp) [![LTP-Core](https://img.shields.io/pypi/v/ltp-core?label=LTP-Core)](https://pypi.org/project/ltp-core)   [![LTP-Extension](https://img.shields.io/pypi/v/ltp-extension?label=LTP-Extension)](https://pypi.org/project/ltp-extension) |
| [Rust](rust/ltp/README.md)           | [![LTP](https://img.shields.io/crates/v/ltp?label=LTP)](https://crates.io/crates/ltp)                                                                                                                                                                                                                                     |

# LTP 4

LTP（Language Technology Platform） 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作。

## 引用

如果您在工作中使用了 LTP，您可以引用这篇论文

```bibtex
@inproceedings{che-etal-2021-n,
    title = "N-{LTP}: An Open-source Neural Language Technology Platform for {C}hinese",
    author = "Che, Wanxiang  and
      Feng, Yunlong  and
      Qin, Libo  and
      Liu, Ting",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.6",
    doi = "10.18653/v1/2021.emnlp-demo.6",
    pages = "42--49",
    abstract = "We introduce N-LTP, an open-source neural language technology platform supporting six fundamental Chinese NLP tasks: lexical analysis (Chinese word segmentation, part-of-speech tagging, and named entity recognition), syntactic parsing (dependency parsing), and semantic parsing (semantic dependency parsing and semantic role labeling). Unlike the existing state-of-the-art toolkits, such as Stanza, that adopt an independent model for each task, N-LTP adopts the multi-task framework by using a shared pre-trained model, which has the advantage of capturing the shared knowledge across relevant Chinese tasks. In addition, a knowledge distillation method (Clark et al., 2019) where the single-task model teaches the multi-task model is further introduced to encourage the multi-task model to surpass its single-task teacher. Finally, we provide a collection of easy-to-use APIs and a visualization tool to make users to use and view the processing results more easily and directly. To the best of our knowledge, this is the first toolkit to support six Chinese NLP fundamental tasks. Source code, documentation, and pre-trained models are available at https://github.com/HIT-SCIR/ltp.",
}
```

**参考书：**
由哈工大社会计算与信息检索研究中心（HIT-SCIR）的多位学者共同编著的《[自然语言处理：基于预训练模型的方法](https://item.jd.com/13344628.html)
》（作者：车万翔、郭江、崔一鸣；主审：刘挺）一书现已正式出版，该书重点介绍了新的基于预训练模型的自然语言处理技术，包括基础知识、预训练词向量和预训练模型三大部分，可供广大LTP用户学习参考。

### 更新说明

- 4.2.0
  - \[结构性变化\] 将 LTP 拆分成 2 个部分，维护和训练更方便，结构更清晰
    - \[Legacy 模型\] 针对广大用户对于**推理速度**的需求，使用 Rust 重写了基于感知机的算法，准确率与 LTP3 版本相当，速度则是 LTP v3 的 **3.55** 倍，开启多线程更可获得 **17.17** 倍的速度提升，但目前仅支持分词、词性、命名实体三大任务
    - \[深度学习模型\] 即基于 PyTorch 实现的深度学习模型，支持全部的6大任务（分词/词性/命名实体/语义角色/依存句法/语义依存）
  - \[其他改进\] 改进了模型训练方法
    - \[共同\] 提供了训练脚本和训练样例，使得用户能够更方便地使用私有的数据，自行训练个性化的模型
    - \[深度学习模型\] 采用 hydra 对训练过程进行配置，方便广大用户修改模型训练参数以及对 LTP 进行扩展（比如使用其他包中的 Module）
  - \[其他变化\] 分词、依存句法分析 (Eisner) 和 语义依存分析 (Eisner) 任务的解码算法使用 Rust 实现，速度更快
  - \[新特性\] 模型上传至 [Huggingface Hub](https://huggingface.co/LTP)，支持自动下载，下载速度更快，并且支持用户自行上传自己训练的模型供LTP进行推理使用
  - \[破坏性变更\] 改用 Pipeline API 进行推理，方便后续进行更深入的性能优化（如SDP和SDPG很大一部分是重叠的，重用可以加快推理速度），使用说明参见[Github快速使用部分](https://github.com/hit-scir/ltp)
- 4.1.0
  - 提供了自定义分词等功能
  - 修复了一些bug
- 4.0.0
  - 基于Pytorch 开发，原生 Python 接口
  - 可根据需要自由选择不同速度和指标的模型
  - 分词、词性、命名实体、依存句法、语义角色、语义依存6大任务

## 快速使用

### [Python](python/interface/README.md)

```bash
pip install -U ltp ltp-core ltp-extension -i https://pypi.org/simple # 安装 ltp
```

**注：** 如果遇到任何错误，请尝试使用上述命令重新安装 ltp，如果依然报错，请在 Github issues 中反馈。

```python
import torch
from ltp import LTP

ltp = LTP("LTP/small")  # 默认加载 Small 模型

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
# 使用字典格式作为返回结果
print(output.cws)  # print(output[0]) / print(output['cws']) # 也可以使用下标访问
print(output.pos)
print(output.sdp)

# 使用感知机算法实现的分词、词性和命名实体识别，速度比较快，但是精度略低
ltp = LTP("LTP/legacy")
# cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "ner"]).to_tuple() # error: NER 需要 词性标注任务的结果
cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner"]).to_tuple()  # to tuple 可以自动转换为元组格式
# 使用元组格式作为返回结果
print(cws, pos, ner)
```

**[详细说明](python/interface/docs/quickstart.rst)**

### [Rust](rust/ltp/README.md)

```rust
use std::fs::File;
use itertools::multizip;
use ltp::{CWSModel, POSModel, NERModel, ModelSerde, Format, Codec};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let file = File::open("data/legacy-models/cws_model.bin")?;
  let cws: CWSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
  let file = File::open("data/legacy-models/pos_model.bin")?;
  let pos: POSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
  let file = File::open("data/legacy-models/ner_model.bin")?;
  let ner: NERModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;

  let words = cws.predict("他叫汤姆去拿外衣。")?;
  let pos = pos.predict(&words)?;
  let ner = ner.predict((&words, &pos))?;

  for (w, p, n) in multizip((words, pos, ner)) {
    println!("{}/{}/{}", w, p, n);
  }

  Ok(())
}
```

## 模型性能以及下载地址

|                  深度学习模型                   |  分词   |  词性   | 命名实体  | 语义角色  | 依存句法  | 语义依存  | 速度(句/S) |
| :---------------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
|  [Base](https://huggingface.co/LTP/base)  | 98.7  | 98.5  | 95.4  | 80.6  | 89.5  | 75.2  |  39.12  |
| [Base1](https://huggingface.co/LTP/base1) | 99.22 | 98.73 | 96.39 | 79.28 | 89.57 | 76.57 |  --.--  |
| [Base2](https://huggingface.co/LTP/base2) | 99.18 | 98.69 | 95.97 | 79.49 | 90.19 | 76.62 |  --.--  |
| [Small](https://huggingface.co/LTP/small) | 98.4  | 98.2  | 94.3  | 78.4  | 88.3  | 74.7  |  43.13  |
|  [Tiny](https://huggingface.co/LTP/tiny)  | 96.8  | 97.1  | 91.6  | 70.9  | 83.8  | 70.1  |  53.22  |

|                    感知机算法                    |  分词   |  词性   | 命名实体  | 速度(句/s)  |             备注             |
| :-----------------------------------------: | :---: | :---: | :---: | :------: | :------------------------: |
| [Legacy](https://huggingface.co/LTP/legacy) | 97.93 | 98.41 | 94.28 | 21581.48 | [性能详情](rust/ltp/README.md) |

**注：感知机算法速度为开启16线程速度**

## 构建 Wheel 包

```shell script
make bdist
```

## 其他语言绑定

**感知机算法**

- [Rust](rust/ltp)
- [C/C++](rust/ltp-cffi)

**深度学习算法**

- [Rust](https://github.com/HIT-SCIR/libltp/tree/master/ltp-rs)
- [C++](https://github.com/HIT-SCIR/libltp/tree/master/ltp-cpp)
- [Java](https://github.com/HIT-SCIR/libltp/tree/master/ltp-java)

## 作者信息

- 冯云龙 \<\<[ylfeng@ir.hit.edu.cn](mailto:ylfeng@ir.hit.edu.cn)>>

## 开源协议

1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。
2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。
3. 凡涉及付费问题，请发邮件到 car@ir.hit.edu.cn 洽商。
4. 如果您在 LTP 基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”.
   同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。
