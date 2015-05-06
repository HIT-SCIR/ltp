#!/usr/bin/env python
# -*- coding: utf-8 -*-
print """#ifndef __LTP_STRUTILS_SENTSPLIT_TAB__
#define __LTP_STRUTILS_SENTSPLIT_TAB__

#pragma warning(disable: 4309)
"""
double_periods = [u"。”", u"！”", u"？”", u"；”", u"……”", u"？！"]
single_periods = [u"。", u"！", u"？", u"；", u"……"]

print "static const int __double_periods_utf8_size__ = %d;" % len(double_periods)
print "static const char* __double_periods_utf8_buff__[] = {"
for i, k in enumerate(double_periods):
    print "\"%s\"," % k.encode("utf-8")
print "};"

print "static const int __single_periods_utf8_size__ = %d;" % len(single_periods)
print "static const char* __single_periods_utf8_buff__[] = {"
for i, k in enumerate(single_periods):
    print "\"%s\"," % k.encode("utf-8")
print "};"

print "#endif   // end for __LTP_STRUTILS_SENTSPLIT_TAB__"
