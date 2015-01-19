新闻
=====

语言技术平台3.2.0版 发布

* [增加] 分词模块增量模型训练工具。使用户可以在语言技术平台基线模型的基础之上增加训练数据，从而获得特定领域性能更好的模型。
* [修改] Boost.Regex到1.56.0，由于旧版本Boost.Regex的 `match_results` 类存在竞争问题，这一修改修复了 `multi_cws_cmdline` 随机出错的问题。
* [修改] 使自动化测试脚本支持Windows运行以及多线程测试
* [修改] 将原 `examples` 文件夹下的示例文件转移到 `test` 文件夹下并纳入语言技术平台的编译链
* [测试] 新版语言技术平台通过 `cygwin` 编译测试
* [测试] 多线程程序 `multi_ltp_test` ， `multi_cws_cmdline` 以及 `multi_pos_cmdline` 在Windows通过测试
