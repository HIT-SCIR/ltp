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

## Benchmark Compare with Jieba

- File Size: 33.85 MB
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) | Speed(KB/s) |
| -------------- | ------: | ----------: |
| Jieba cut      |   35.29 |      982.49 |
| LTP legacy(1)  |   36.33 |      954.08 |
| LTP legacy(2)  |   19.41 |     1786.08 |
| LTP legacy(4)  |   10.74 |     3228.71 |
| LTP legacy(8)  |    7.07 |     4904.05 |
| LTP legacy(16) |    5.89 |     5880.19 |

## Benchmark Pipeline

- File Size: 33.85 MB / 305041 lines
- Hard Ware:
  - MacBook Pro (16-inch, 2019)
  - 处理器: 2.6 GHz 六核Intel Core i7
  - 内存: 16 GB 2667 MHz DDR4
  - 图形卡: Intel UHD Graphics 630 1536 MB

| Algorithm      | Time(s) | Speed(KB/s) |
| -------------- | ------: | ----------: |
| LTP legacy(1)  |   90.66 |      382.33 |
| LTP legacy(2)  |   49.43 |      701.23 |
| LTP legacy(4)  |   27.98 |     1238.76 |
| LTP legacy(8)  |   20.11 |     1723.72 |
| LTP legacy(16) |   16.99 |     2040.26 |
