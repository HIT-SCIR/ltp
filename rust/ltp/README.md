![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)

| Language                             | version                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Python](python/interface/README.md) | [![LTP](https://img.shields.io/pypi/v/ltp?label=LTP)](https://pypi.org/project/ltp) [![LTP-Core](https://img.shields.io/pypi/v/ltp-core?label=LTP-Core)](https://pypi.org/project/ltp-core)   [![LTP-Extension](https://img.shields.io/pypi/v/ltp-extension?label=LTP-Extension)](https://pypi.org/project/ltp-extension) |
| [Rust](rust/ltp/README.md)           | [![LTP](https://img.shields.io/crates/v/ltp?label=LTP)](https://crates.io/crates/ltp)                                                                                                                                                                                                                                     |

# LTP For Rust

传统机器学习方法（LTP 3）实现的 CWS / POS / NER 算法。

| method | ltp 3.0(c++) | ap(1) | ap(8) | pa    | pa-i(0.5) | pa-ii(0.5) |
| ------ | ------------ | ----- | ----- | ----- | --------- | ---------- |
| cws    | 97.83        | 97.93 | 97.67 | 97.90 | 97.90     | 97.93      |
| pos    | 98.35        | 98.41 | 98.30 | 98.39 | 98.39     | 98.38      |
| ner    | 94.17        | 94.28 | 93.42 | 94.02 | 94.06     | 93.95      |

## 快速使用

```rust
use std::fs::File;
use apache_avro::Codec;
use itertools::multizip;
use ltp::{CWSModel, POSModel, NERModel, ModelSerde, Format};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data/legacy-models/cws_model.bin")?;
    let cws: CWSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
    let file = File::open("data/legacy-models/pos_model.bin")?;
    let pos: POSModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;
    let file = File::open("data/legacy-models/ner_model.bin")?;
    let ner: NERModel = ModelSerde::load(file, Format::AVRO(Codec::Deflate))?;

    let words = cws.predict("他叫汤姆去拿外衣。");
    let pos = pos.predict(&words);
    let ner = ner.predict((&words, &pos));

    for (w, p, n) in multizip((words, pos, ner)) {
        println!("{}/{}/{}", w, p, n);
    }

    Ok(())
}
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

## Benchmark Compare

- File Size: 33.85 MB / 305041 lines
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) |       Speed(KB/s) |
|----------------|--------:|------------------:|
| Jieba cut      |   35.29 |            982.49 |
| Pkuseg         |  315.91 |            109.72 |
| Thulac         |  720.19 |             48.13 |
| Thulac(Fast)   |   30.59 |           1133.21 |
| LTP 3          |   76.82 |            451.20 |
| LTP legacy(1)  |   36.33 |            954.08 |
| LTP legacy(2)  |   19.41 |           1786.08 |
| LTP legacy(4)  |   10.74 |           3228.71 |
| LTP legacy(8)  |    7.07 |           4904.05 |
| LTP legacy(16) |    5.89 |           5880.19 |

**注：括号内为线程数量**

## Benchmark Pipeline (CWS/POS/NER)

- File Size: 33.85 MB / 305041 lines
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) | Speed(KB/s) |
| -------------- |--------:|------------:|
| LTP 3          |  226.40 |      153.10 |
| LTP legacy(1)  |   90.66 |      382.33 |
| LTP legacy(2)  |   49.43 |      701.23 |
| LTP legacy(4)  |   27.98 |     1238.76 |
| LTP legacy(8)  |   20.11 |     1723.72 |
| LTP legacy(16) |   16.99 |     2040.26 |

**注：括号内为线程数量**
