# -*- coding: utf-8 -*-
from socket import *
from ltp_interface import *
import datetime, time, sys, os
ErrorInfoMap = {
    0: "OK",
    -1: "SplitSentence error(分句错误)",
    -2: "Word Segmentation and POS error(分词词性标注错误)",
    -3: "Name Entity recognition(NE) error(命名实体识别错误)",
    -4: "Word Sense Disambiguation(WSD) error(词义消歧错误)",
    -5: "Dependency Parser error(句法分析错误)",
    -6: "Shallow Semantic Role Labelling error(语义分析错误)",
    -7: "Text Classification error(文本分类错误)",
    -8: "Document Summarization error(自动文摘错误)",
    -9: "Coreference Resolution error(指代消解错误)",
    -10: "Save DOM Tree error(各个模块处理完成之后存储DOM出错)",
    -11: "Create DOM Tree from txt file error(从xml文件创建DOM树错误)",
    -12: "Create DOM Tree from txt file error(从txt文件创建DOM树错误)",      
}

#-11	从xml文件创建DOM树错误
#-12	从txt文件创建DOM树错误
#-1	分句错误
#-2	分词词性标注错误
#-3	命名实体识别错误
#-4	词义消歧错误
#-5	句法分析错误
#-6	语义分析错误
#-7	文本分类错误
#-8	自动文摘错误
#-9	指代消解错误
#-10	各个模块处理完成之后存储DOM出错

ret = py_main2('test_start.txt', 'test_start.xml')
if ErrorInfoMap.has_key(ret):
    print 'test_start: ', ErrorInfoMap[ret]
else:
    print 'test_start: 其他未知错误'


log_f = open('ltp_server.log', 'a')

myHost = ''
myPort = 50010

sockobj = socket(AF_INET, SOCK_STREAM)
sockobj.bind((myHost, myPort))
sockobj.listen(5)

print 'bind port: ', myPort

while True:
    connection, address = sockobj.accept()
    print 'ltp v2.0 Demo Server connected by', address
    
    start_time = datetime.datetime.now()
    print 'start time: ', start_time
    data = connection.recv(0xFF) 
    if data != 'START':
        continue
    try:
        in_txt_file_name = 'ltp_in.txt'
        out_xml_file_name = 'ltp_out.xml'

        ret = py_main2(in_txt_file_name, out_xml_file_name)
        ClearDOM()
        if ErrorInfoMap.has_key(ret):
            data = ErrorInfoMap[ret]
        else:
            data = '其他未知错误'
    except Exception, msg:
        data = "ltp server Exception: " + repr(msg)

    #inf = open(out_xml_file_name, 'r')
    #data = inf.read()
    #inf.close()
    
    connection.send(data)
    connection.close()

    if data != 'OK':
        print data
        localtime = time.asctime(time.localtime(time.time()))
        log_f.write(localtime + ' : ' + data + '\n')
        #log_f.close()
        #sys.exit(-1)
        #break

    end_time = datetime.datetime.now()
    print 'Process Time: %d seconds' % (end_time - start_time).seconds

log_f.close()
