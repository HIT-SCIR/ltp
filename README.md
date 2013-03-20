# Ltp: Language Technology Platform

[![Build Status](https://travis-ci.org/HIT-SCIR/ltp.png?branch=master)](https://travis-ci.org/HIT-SCIR/ltp)

简介
----

__[语言技术平台（Language Technology Platform，LTP）](http://ir.hit.edu.cn/ltp/)__是[哈工大社会计算与信息检索研究中心](http://ir.hit.edu.cn/)历时十年开发的一整套中文语言处理系统。LTP制定了基于XML的语言处理结果表示，并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块（包括词法、句法、语义等6项中文处理核心技术），以及基于动态链接库（Dynamic Link Library, DLL）的应用程序接口，可视化工具，并且能够以网络服务（Web Service）的形式进行使用。

从2006年9月5日开始该平台对外免费共享目标代码，截止目前，已经有国内外400多家研究单位共享了LTP，也有国内外多家商业公司购买了LTP，用于实际的商业项目中。2010年12月获得中国中文信息学会颁发的行业最高奖项：“钱伟长中文信息处理科学技术奖”一等奖。

2011年6月1日，为了与业界同行共同研究和开发中文信息处理核心技术，我中心正式将LTP开源。

编译
----

2013年3月以后，为适应跨平台编译，LTP从Automake改为使用CMake编译，编译时请注意对应版本。

__2.2.0之后__

1. 将ltp_data.zip压缩包解压至项目文件夹下
2. 配置
```
./configure
```
3. 编译
```
make
```

编译后会在bin/下产生两个可执行程序`ltp_test`和`ltp_test_xml`，同时会在lib/下产生各组件的静态链接库。

__2.2.0之前__

1. 将ltp_data.zip压缩包解压至项目文件夹下
2. 配置
```
./configure
```
3. 编译
```
make
```

编译后会在src/test/下产生两个可执行程序`ltp_test`和`ltp_test_xml`。

模型
----

由于模型文件`ltp_data.zip`不适合进行版本控制，现已经将`ltp_data.zip`转移到[这里](http://ir.hit.edu.cn/ltp/program/ltp_data.zip)。

