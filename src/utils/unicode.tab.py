#!/usr/bin/env python
import unicodedata
import sys
codes = [i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P')]

print "#ifndef __LTP_UTILS_UNICODE_TAB__"
print "#define __LTP_UTILS_UNICODE_TAB__"

print "const static unsigned UNICODE_PUNCTUATION[]= {"
nr = 0
for i, c in enumerate(codes):
    if i == 0:
        print "  %s," % hex(c),
        nr += 1
    elif codes[i-1]+1 != codes[i]:
        print "  %s," % hex(c),
        nr += 1

    if i+1 == len(codes):
        print "%s," % hex(c)
        nr += 1
    elif codes[i]+1 != codes[i+1]:
        print "%s," % hex(c)
        nr += 1
print "};"

print "const static int UNICODE_PUNCTUATION_SIZE=%d;"% nr

print "#endif   //  end for __LTP_UTILS_UNICODE_TAB__"
