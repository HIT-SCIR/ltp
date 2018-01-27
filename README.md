# LTP: Language Technology Platform

[![Join the chat at https://gitter.im/HIT-SCIR/ltp](https://badges.gitter.im/HIT-SCIR/ltp.svg)](https://gitter.im/HIT-SCIR/ltp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/HIT-SCIR/ltp.svg?branch=master)](https://travis-ci.org/HIT-SCIR/ltp)
[![Build status](https://ci.appveyor.com/api/projects/status/yewlrwa7w85kghwb/branch/master?svg=true)](https://ci.appveyor.com/project/Oneplus/ltp/branch/master)
[![Documentation Status](https://readthedocs.org/projects/ltp/badge/?version=latest)](https://readthedocs.org/projects/ltp/?badge=latest)

新闻
----

语言技术平台Docker版本发布
* 随着LTP版本升级引入第三方库导致目前系统在不同环境下编译遇到各种复杂问题。由于我们的人力确实不足，这些问题的解决可能是一个缓慢的过程。我们现发布了ltp 3.4.0 docker版本。
* 版本已经包含编译完成的程序和模型，无需额外下载安装其他任何软件。
* 请参考 [使用 Docker 中的LTP](http://ltp.ai/docs/install.html#) 进行使用。

语言云全面改用`HTTPS`协议
* 由于安全需求，目前改仓库的云接口平台 [语言技术平台云](https://www.ltp-cloud.com/) 已经全面换用`HTTPS`协议访问。
* 之前使用云平台接口的用户，请切换到`HTTPS`协议访问、调用接口。
* 如果遇到平台网页不能打开的情况，可以清理浏览器缓存解决。
* 在迁移期间导致部分新注册用户`apikey`不能使用情况，请联系管理员。

语言技术平台官网上线
* [语言技术平台官网](http://ltp.ai/)近期上线。
* 可访问官网 [查看文档](http://ltp.ai/docs/index.html) 、 [下载模型](http://ltp.ai/download.html) 、 [体验在线Demo](http://ltp.ai/demo.html) 。

模型切换到七牛源
* 最近很多用户反映使用百度云下载模型非常缓慢，现已切换到七牛云，请访问[该链接](http://ltp.ai/download.html)选择版本下载。

语言技术平台3.4.0版 发布
* [增加] 新的基于Bi-LSTM的SRL模型
* [增加] 增加了SRL的多线程命令行程序`srl_cmdline`
* [修改] SRL相关的编程接口已经改变，修复了之前内存泄露的相关问题。

语言技术平台 3.3.2 版发布
* [修复] 修复了 3.3.1 版本的一些 bug

语言技术平台 3.3.1 版发布
* [修复] 修复了 3.3.0 版本模型加载的 bug
* [增加] 提供 Windows 下的 `ltp_test` 和 `xxx_cmdline` 二进制下载，无需再手工编译

语言技术平台 3.3.0 版发布
* [增加] 词性标注模型加入微博数据，使得在开放域上的词性标注性能更好(+3.3 precision)
* [增加] 依存句法分析模型加入微博数据，使得在开放域上的句法分析性能更好(+3 UAS)
* [增加] 依存句法分析算法切换到transition-based neural network parser，速度从40 tokens/s提升到8000 tokens/s。同时通过加入聚类特征以及优化训练算法，（在新闻领域）准确率也得到小幅提升(+0.2 UAS)
* [增加] `ltp_test`默认支持多线程，线程数可配置。
* [增加] 新加入子模块命令行程序，`cws_cmdline`，`pos_cmdline`，`par_cmdline`，`ner_cmdline`，使用户可以很容易替换中间模块，从而实现语言分析的组合。
* [修改] 优化了训练套件的交互方式
* [增加] 添加模型验证，单元测试模块。

简介
----

语言技术平台（Language Technology Platform，LTP）是[哈工大社会计算与信息检索研究中心](http://ir.hit.edu.cn/)历时十年开发的一整套中文语言处理系统。LTP制定了基于XML的语言处理结果表示，并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块（包括词法、句法、语义等6项中文处理核心技术），以及基于动态链接库（Dynamic Link Library, DLL）的应用程序接口、可视化工具，并且能够以网络服务（Web Service）的形式进行使用。

从2006年9月5日开始该平台对外免费共享目标代码，截止目前，已经有国内外400多家研究单位共享了LTP，也有国内外多家商业公司购买了LTP，用于实际的商业项目中。

2010年12月获得中国中文信息学会颁发的行业最高奖项：“钱伟长中文信息处理科学技术奖”一等奖。

2011年6月1日，为了与业界同行共同研究和开发中文信息处理核心技术，我中心正式将LTP开源。

2013年9月1日，语言技术平台云端服务"[语言云](http://ltp-cloud.com)"正式上线。

2016年8月，语言技术平台获“黑龙江省科技进步一等奖”。

文档
---

关于LTP的使用，请参考[在线文档](http://ltp.ai/docs/index.html)

模型
---

* [所有版本下载链接](http://ltp.ai/download.html)
* 当前模型版本 3.4.0

其它语言接口
------------
如果您希望在本地使用除C++之外的其他语言调用LTP，我们针对常用语言对LTP进行了封装。

* Python: [pyltp - the python extension for LTP](https://github.com/HIT-SCIR/pyltp)
* Java: [ltp4j - Language Technology Platform for Java](https://github.com/HIT-SCIR/ltp4j)

开源协议
-------

1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。

2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。

3. 凡涉及付费问题，请发邮件到 car@ir.hit.edu.cn 洽商。

4. 如果您在 LTP 基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”，参考文献中加入以下论文： Wanxiang Che, Zhenghua Li, Ting Liu. LTP: A Chinese Language Technology Platform. In Proceedings of the Coling 2010:Demonstrations. 2010.08, pp13-16, Beijing, China. 同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。
