# -*- coding: cp936 -*-

#logfile = file('test_ltp_python_interface_log.txt', 'w')

from ltp_interface import *


#CreateDOMFromString("那里车位很少，停车很麻烦，有时候为了安全还要停在旁边花园里");

CreateDOMFromTxt("test.txt");
SplitSentence()
#IRLAS()
CRFWordSeg()
#PosTag()
#WSD()
#NER()
#Parser()
#GParser()
#SRL()
SaveDOM('test.xml')

'''
print GetWordsFromSentence(0)
print GetNEsFromSentence(0)

sentNum = CountSentenceInDocument()
for i in range(0, sentNum):
    print GetSentence(i)+"\n"
'''

'''
#----------

print CreateDOMFromTxt('test.txt')
#print SaveDOM('test.xml')
#print CreateDOMFromXml('test.xml')
print SplitSentence()
#print IRLAS()
#print NER()
print WSD()
#print GParser()
print SaveDOM('test.xml')
#print SRL()


#----------

print CreateDOMFromString("伊拉克军方官员20日宣布，上周五在巴格达南部地区\
“失踪”的两名美军士兵被当地的反美武装俘虏并且惨遭杀害。20日上午，\
搜救人员在一座变电站附近找到这两名军人的尸体。调查人员表示，有迹象表明，\
这两名美军在死前曾遭到“非常残酷地虐待”。据悉，这两名只有23岁和25岁的美\
军被俘前曾在巴格达南部的公路检查站执勤。武装分子上周五偷袭了该检查站时除\
将上述两人俘虏外，还将另一名美军打死。美军和伊拉克安全部队随后派出了8000多人\
开展了大规模的搜救工作，最终找到了这两名士兵的遗体。")
print IRLAS()
print SaveDOM('test_string.xml')

#----------

CreateDOMFromXml('test.xml')
NER()
GParser()
WSD()
SRL()
SaveDOM('test.xml')
'''
#----------
'''
CreateDOMFromXml('test.xml')
paraNum = CountParagraphInDocument()
print paraNum
for i in range(paraNum):
    sentNum = CountSentenceInParagraph(i)
    print sentNum, '\n-----'
    for j in range(sentNum):
        print CountWordInSentence(i, j)
    break

sentNum = CountSentenceInDocument()
print sentNum, '\n-----'
for i in range(4):
    print CountWordInSentence(i)

#----------

CreateDOMFromXml('test.xml')

sentNum = CountSentenceInDocument()
for i in range(4, 6):
    wordNum = CountWordInSentence(i)
    for j in range(wordNum-10, wordNum):
        print GetWord(i, j),
        print GetPOS(i, j),
        print GetNE(i, j),
        (wsd, explain) = GetWSD(i, j)
        print wsd, explain,
        #print GetWSD(i, j)
        (parent, relate) = GetParse(i, j)
        print parent, relate
        #print GetParse(i, j)

#----------

CreateDOMFromXml('test.xml')
sentNum = CountSentenceInDocument()
for i in range(sentNum):
    word_list = GetWordsFromSentence(i)
    pos_list = GetPOSsFromSentence(i)
    ne_list = GetNEsFromSentence(i)
    wsd_list = GetWSDsFromSentence(i)
    explain_list = GetWSDExplainsFromSentence(i)
    (parent_list, relate_list) = GetParsesFromSentence(i)
    for j in range(len(word_list)):
        logfile.write("%s %s %s %s %s %d %s\n" % (word_list[j],
                                                  pos_list[j],
                                                  ne_list[j],
                                                  wsd_list[j],
                                                  explain_list[j],
                                                  parent_list[j],
                                                  relate_list[j]
                                                  )
                      )
    logfile.write("\n-----\n")

#--------------
    
CreateDOMFromXml('test.xml')
#print GetTextSummary()
#print GetTextClass()
wordNum = CountWordInDocument()
for i in range(wordNum):
    type_list, beg_list, end_list = GetPredArgToWord(i)
    if type_list == None or type_list == []:
        continue
    for (t, b, e) in zip(type_list, beg_list, end_list):
        logfile.write("%s %d %s\n" % (t, b, e))
'''  


#logfile.close()


