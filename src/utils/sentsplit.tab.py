#!/usr/bin/env python
# -*- coding: utf-8 -*-
print """#ifndef __LTP_STRUTILS_SENTSPLIT_TAB__
#define __LTP_STRUTILS_SENTSPLIT_TAB__

#pragma warning(disable: 4309)
"""
_3 = [u"？！”", u"。’”", u"！’”"]
_2 = [u"。”", u"！”", u"？”", u"；”", u"……”", u"？！"]
_1 = [u"。", u"！", u"？", u"；", u"……"]

print "static const int __three_periods_utf8_size__ = %d;" % len(_3)
print "static const char* __three_periods_utf8_buff__[] = {"
for i, k in enumerate(_3):
    print "\"%s\"," % k.encode("utf-8").__repr__()[1:-1]
print "};"

print "static const int __two_periods_utf8_size__ = %d;" % len(_2)
print "static const char* __two_periods_utf8_buff__[] = {"
for i, k in enumerate(_2):
    print "\"%s\"," % k.encode("utf-8").__repr__()[1:-1]
print "};"

print "static const int __one_periods_utf8_size__ = %d;" % len(_1)
print "static const char* __one_periods_utf8_buff__[] = {"
for i, k in enumerate(_1):
    print "\"%s\"," % k.encode("utf-8").__repr__()[1:-1]
print "};"

print "#endif   // end for __LTP_STRUTILS_SENTSPLIT_TAB__"
