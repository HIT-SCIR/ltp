# LTP: Language Technology Platform

[![Build Status](https://travis-ci.org/HIT-SCIR/ltp.svg?branch=master)](https://travis-ci.org/HIT-SCIR/ltp) [![Build status](https://ci.appveyor.com/api/projects/status/yewlrwa7w85kghwb/branch/master?svg=true)](https://ci.appveyor.com/project/Oneplus/ltp/branch/master) [![Documentation Status](https://readthedocs.org/projects/ltp/badge/?version=latest)](https://readthedocs.org/projects/ltp/?badge=latest)

新闻
----

语言技术平台3.3.1版 发布
* [修复] 修复了3.3.0版本模型加载的bug
* [增加] 提供 Windows 下的`ltp_test`和`xxx_cmdline`二进制下载，无需再手工编译

语言技术平台3.3.0版 发布
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

文档
---

关于LTP的使用，请参考[在线文档](http://ltp.readthedocs.org/zh_CN/latest/)

模型
---

* [百度云](http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569)
* 当前模型版本3.3.1

开源协议
-------

1. 语言技术平台面向国内外大学、中科院各研究所以及个人研究者免费开放源代码，但如上述机构和个人将该平台用于商业目的（如企业合作项目等）则需要付费。

2. 除上述机构以外的企事业单位，如申请使用该平台，需付费。

3. 凡涉及付费问题，请发邮件到car@ir.hit.edu.cn洽商。

4. 如果您在LTP基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了哈工大社会计算与信息检索研究中心研制的语言技术平台（LTP）”，参考文献中加入以下论文： Wanxiang Che, Zhenghua Li, Ting Liu. LTP: A Chinese Language Technology Platform. In Proceedings of the Coling 2010:Demonstrations. 2010.08, pp13-16, Beijing, China. 同时，发信给car@ir.hit.edu.cn，说明发表论文或申报成果的题目、出处等。
