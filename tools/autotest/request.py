#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import urllib, urllib2
from optparse import OptionParser

uri_base = "http://127.0.0.1:12345/ltp"

def EncodingError():
    '''
    Construct an request in non-UTF8 encoding.

    Expect get a 400 error, with reason 
    '''
    data = {
            's': '我爱北京天安门。'.decode('utf-8').encode('gbk'),
            'x': 'n',
            't': 'all'}
    request  = urllib2.Request(uri_base)
    params   = urllib.urlencode(data)
    try:
        response = urllib2.urlopen(request, params)
        content  = response.read().strip()
        print content
        return False
    except urllib2.HTTPError, e:
        if e.code == 400 and e.reason == "ENCODING NOT IN UTF8":
            return True
        else:
            return False


def XMLError():
    ''' Construct a request specify XML but actually is a plain string. '''
    data = {
            's': '我爱北京天安门。',
            'x': 'y',
            't': 'all'}
    request  = urllib2.Request(uri_base)
    params   = urllib.urlencode(data)

    try:
        response = urllib2.urlopen(request, params)
        content  = response.read().strip()
        return False
    except urllib2.HTTPError, e:
        if e.code == 400 and e.reason == "BAD XML FORMAT":
            return True
        else:
            return False

def XMLError2():
    ''' Construct a request specify XML but actually is a plain string. '''
    data = {
            's': '<xml4nlp><doc></doc></xml4nlp>',
            'x': 'y',
            't': 'all'}
    request  = urllib2.Request(uri_base)
    params   = urllib.urlencode(data)

    try:
        response = urllib2.urlopen(request, params)
        content  = response.read().strip()
        return False
    except urllib2.HTTPError, e:
        if e.code == 400 and e.reason == "BAD XML FORMAT":
            return True
        else:
            return False

def XMLError3():
    ''' Construct a request specify XML but actually is a plain string. '''
    data = {
            's': '<xml4nlp><doc><para></para></doc></xml4nlp>',
            'x': 'y',
            't': 'all'}
    request  = urllib2.Request(uri_base)
    params   = urllib.urlencode(data)

    try:
        response = urllib2.urlopen(request, params)
        content  = response.read().strip()
        return False
    except urllib2.HTTPError, e:
        if e.code == 400 and e.reason == "BAD XML FORMAT":
            return True
        else:
            return False


def NormalTest():
    '''
    Construct a legal request
    '''
    data = {
            's': '我爱北京天安门。',
            'x': 'n',
            't': 'all'}

    try:
        request  = urllib2.Request(uri_base)
        params   = urllib.urlencode(data)
        response = urllib2.urlopen(request, params)
        content  = response.read().strip()
        return True
    except:
        return False


def Request(text):
    data = {
            's': text,
            'x': 'n',
            't': 'all'}

    request  = urllib2.Request(uri_base)
    params   = urllib.urlencode(data)
    response = urllib2.urlopen(request, params)
    content  = response.read().strip()
    return content


if __name__=="__main__":
    usage = "perform ltp-request-test"
    optparser = OptionParser(usage)
    optparser.add_option("--case", dest="error_cases", action="store_true",
                         default=False, help="specify case test")
    optparser.add_option("--file", dest="filename", help="specify the file")
    opts, args = optparser.parse_args()

    if opts.error_cases:
        def TEST(function, name):
            result = "Passed" if function() else "Failed"
            print "%16s : %s" % (name, result)

        TEST(NormalTest, "Normal TEST")
        TEST(EncodingError, "Encoding ERROR")
        TEST(XMLError, "XML ERROR")
        TEST(XMLError2, "XML ERROR 2")
        TEST(XMLError3, "XML ERROR 3")
    else:
        try:
            fp=open(opts.filename, "r")
        except:
            print >> sys.stderr, "Failed to open file, use stdin instead"
            fp=sys.stdin

        for line in fp:
            try:
                print Request(line.strip())
            except Exception, e:
                print e
