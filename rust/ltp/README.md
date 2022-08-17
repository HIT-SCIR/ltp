![CODE SIZE](https://img.shields.io/github/languages/code-size/HIT-SCIR/ltp)
![CONTRIBUTORS](https://img.shields.io/github/contributors/HIT-SCIR/ltp)
![LAST COMMIT](https://img.shields.io/github/last-commit/HIT-SCIR/ltp)

| Language | version                                                                                       |
| -------- | --------------------------------------------------------------------------------------------- |
| Python   | [![LTP](https://img.shields.io/pypi/v/ltp?label=LTP4%20ALPHA)](https://pypi.org/project/ltp)  |
| Rust     | [![LTP](https://img.shields.io/crates/d/ltp?label=LTP%20Alpha)](https://crates.io/crates/ltp) |

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
    - [ ] 支持自定义词典
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

## Benchmark Compare with Jieba

- File Size: 33.85 MB / 305041 lines
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) | Speed(KB/s) |
| -------------- | ------: | ----------: |
| Jieba cut      |   35.29 |      982.49 |
| LTP legacy(1)  |   58.64 |      591.04 |
| LTP legacy(2)  |   30.64 |     1131.30 |
| LTP legacy(4)  |   17.14 |     2022.50 |
| LTP legacy(8)  |   11.39 |     3044.15 |
| LTP legacy(16) |    9.88 |     3508.93 |

## Benchmark Pipeline

- File Size: 33.85 MB / 305041 lines
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) | Speed(KB/s) |
| -------------- | ------: | ----------: |
| LTP legacy(1)  |  139.70 |      248.11 |
| LTP legacy(2)  |   75.18 |      461.04 |
| LTP legacy(4)  |   42.96 |      806.76 |
| LTP legacy(8)  |   29.34 |     1181.39 |
| LTP legacy(16) |   26.28 |     1318.84 |
