# -*- coding: cp936 -*-
"""
This file provides all the python interfaces of LTP.
To use LTP in python project, you need the following components:
    data files: in ltp_data/
    configure files: *.conf
    dll files: (especially ltp_dll_for_python.dll)
    visualization: nlp_style.xsl
"""

"""
LTP is a language platform based on XML presentation. So all operations are done in DOM.
For now, LTP is oriented to single document.
A document is presented as a DOM, which can be saved as an XML file. The XML format defined in LTP is as following:

---------------
<?xml version="1.0" encoding="gb2312" ?>
<?xml-stylesheet type="text/xsl" href="nlp_style.xsl" ?>
<xml4nlp>
    <note sent="y" word="y" pos="y" ne="y" parser="y" wsd="y" srl="y" class="y" sum="y" cr="y" />
    <doc>
        <para id="0">
            <sent id="0" cont="伊拉克军方官员20日宣布，...">
                <word id="0" cont="伊拉克" pos="ns" ne="O" parent="1" relate="ATT" wsd="Di02" wsdexp="国家_行政区划" />
                ...
                <word id="4" cont="宣布" pos="v" ne="O" parent="-1" relate="HED" wsd="Hc11" wsdexp="召集_宣布_下令">
                    <arg id="0" type="Arg0" beg="0" end="2" />
                    <arg id="1" type="ArgM-TMP" beg="3" end="3" />
                    <arg id="2" type="Arg1" beg="6" end="28" />
                </word>
                ...
            <sent id="1" cont="20日上午，搜救人员在一座变电站附近找到这两名军人的尸体。">
            ...
        </para>
        ...
    </doc>
</xml4nlp>        
---------------

As we can see from above:
A document after fully processed (Split Sentence, Word Segment, POS, NE Recognition, Word Sense Disambiguity, Parser,
Semantic Role Labelling, Text Classify, Text Summary, Coreference Resolution), is organized as following:

each <doc> is composed of several <para>
each <para> is composed of several <sent>
each <sent> is composed of several <word>
each <word> has several attributes to represent the POS, NE, Parser, WSD info of this word.
each <word> has zero or several <arg>, which reprsents the SRL info of this word.

Note:
While, the "beg" and "end" attibutes in <arg> is the local word index in current sentence.

"""

"""
restype: If there is no special declaration, the return value is int. 0 represent a success, while -1 means a failure.
argtypes: There are several interfaces which accept variable arguments.
"""

from ctypes import *
ltp_dll = CDLL("__ltp_dll_for_python.dll")

def py_main2(inFileName, xmlFileName):
    return ltp_dll.py_main2(inFileName, xmlFileName, 'ltp_modules_to_do.conf')

#print py_main2('test.txt', 'test__.xml')

def CreateDOMFromTxt(fileName):
    return ltp_dll.CreateDOMFromTxt(fileName)

def CreateDOMFromXml(fileName):
    return ltp_dll.CreateDOMFromXml(fileName)

def CreateDOMFromString(strText):
    return ltp_dll.CreateDOMFromString(strText)

def SaveDOM(fileName):
    return ltp_dll.SaveDOM(fileName)

def ClearDOM():
    return ltp_dll.ClearDOM()

def SplitSentence():
    return ltp_dll.SplitSentence()

def WordSegment():
    """
    Word Segment
    """
    return ltp_dll.WordSegment()

def CRFWordSeg():
    """
    CRF-based Word Segment
    """
    return ltp_dll.CRFWordSeg()

def PosTag():
    """
    POS Tagging
    """
    return ltp_dll.PosTag()

def IRLAS():
    """
    Word Segment and POS
    """
    return ltp_dll.IRLAS()

def NER():
    """
    NE Recognition
    """
    return ltp_dll.NER()

def WSD():
    """
    Word Sense Disambiguity
    """
    return ltp_dll.WSD()

def Parser():
    return ltp_dll.Parser()

def GParser():
    return ltp_dll.GParser()

def SRL():
    return ltp_dll.SRL()

#
# Counting
#
def CountParagraphInDocument():
    return ltp_dll.CountParagraphInDocument()

def CountSentenceInParagraph(paraIdx):
    return ltp_dll.CountSentenceInParagraph(paraIdx)

def CountSentenceInDocument():
    return ltp_dll.CountSentenceInDocument()

def CountWordInSentence(*args):
    if len(args) == 2:
        paraIdx, sentIdx = args
        return ltp_dll.CountWordInSentence_p(paraIdx, sentIdx)
    elif len(args) == 1:
        globalSentIdx, = args
        return ltp_dll.CountWordInSentence(globalSentIdx)
    else:
        return -1

def CountWordInDocument():
    return ltp_dll.CountWordInDocument()

ltp_dll.GetParagraph.restype = c_char_p
def GetParagraph(paraIdx):
    """
    restype: the paragraph text (str)
    """
    return ltp_dll.GetParagraph(paraIdx)

ltp_dll.GetSentence_p.restype = c_char_p
ltp_dll.GetSentence.restype = c_char_p
def GetSentence(*args):
    """
    restype: the sentence text (str)
    argtypes:
        para idx in document, sent idx in para (int, int)
        or
        global sent idx in document (int)        
    """
    if len(args) == 2:
        paraIdx, sentIdx = args
        return ltp_dll.GetSentence_p(paraIdx, sentIdx)
    elif len(args) == 1:
        globalSentIdx, = args
        return ltp_dll.GetSentence(globalSentIdx)
    else:
        return None

ltp_dll.GetWord_p_s.restype = c_char_p
ltp_dll.GetWord_s.restype = c_char_p
ltp_dll.GetWord.restype = c_char_p
def GetWord(*args):
    """
    restype: the word text (str)
    argtypes:
        para idx in document, sent idx in para, word idx in sent (int, int, int)   
        or
        global sent idx in document, word idx in sent (int, int)
        or
        global word idx in document (int)

    The following functions is similar as this.
    """
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        return ltp_dll.GetWord_p_s(paraIdx, sentIdx, wordIdx)
    elif len(args) == 2:
        gSentIdx, wordIdx = args
        return ltp_dll.GetWord_s(gSentIdx, wordIdx)
    elif len(args) == 1:
        gWordIdx, = args
        return ltp_dll.GetWord(gWordIdx)
    else:
        return None

ltp_dll.GetPOS_p_s.restype = c_char_p
ltp_dll.GetPOS_s.restype = c_char_p
ltp_dll.GetPOS.restype = c_char_p
def GetPOS(*args):
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        return ltp_dll.GetPOS_p_s(paraIdx, sentIdx, wordIdx)
    elif len(args) == 2:
        gSentIdx, wordIdx = args
        return ltp_dll.GetPOS_s(gSentIdx, wordIdx)
    elif len(args) == 1:
        gWordIdx, = args
        return ltp_dll.GetPOS(gWordIdx)
    else:
        return None

ltp_dll.GetNE_p_s.restype = c_char_p
ltp_dll.GetNE_s.restype = c_char_p
ltp_dll.GetNE.restype = c_char_p
def GetNE(*args):
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        return ltp_dll.GetNE_p_s(paraIdx, sentIdx, wordIdx)
    elif len(args) == 2:
        gSentIdx, wordIdx = args
        return ltp_dll.GetNE_s(gSentIdx, wordIdx)
    elif len(args) == 1:
        gWordIdx, = args
        return ltp_dll.GetNE(gWordIdx)
    else:
        return None

def GetWSD(*args):
    """
    restype: code and explain in TongYiCiCiLin (str, str)
    """
    wsd = c_char_p()
    explain = c_char_p()
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        if 0 == ltp_dll.GetWSD_p_s(byref(wsd), byref(explain), paraIdx, sentIdx, wordIdx):
            return (wsd.value, explain.value)

    elif len(args) == 2:
        gSentIdx, wordIdx = args
        if 0 == ltp_dll.GetWSD_s(byref(wsd), byref(explain), gSentIdx, wordIdx):
            return (wsd.value, explain.value)

    elif len(args) == 1:
        gWordIdx, = args
        if (0 == ltp_dll.GetWSD(byref(wsd), byref(explain), gWordIdx)):
            return (wsd.value, explain.value)
    return(None, None)

def GetParse(*args):
    """
    restype: parent's word idx in sent and relation type (int, str)
    """
    parent = c_int()
    relate = c_char_p()
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        if (0 == ltp_dll.GetParse_p_s(byref(parent), byref(relate), paraIdx, sentIdx, wordIdx)):
            return (parent.value, relate.value)

    elif len(args) == 2:
        gSentIdx, wordIdx = args
        if (0 == ltp_dll.GetParse_s(byref(parent), byref(relate), gSentIdx, wordIdx)):
            return (parent.value, relate.value)

    elif len(args) == 1:
        gWordIdx, = args
        if (0 == ltp_dll.GetParse(byref(parent), byref(relate), gWordIdx)):
            return (parent.value, relate.value)
    return(None, None)

def GetWordsFromSentence(*args):
    """
    restype: the word list of the sentence (list[str,])
    argtypes:
        para idx in document, sent idx in para (int, int)   
        or
        global sent idx in document (int)

    The following functions is similar as this.
    """
    word_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        word_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetWordsFromSentence_p(word_arr, wordNum, paraIdx, sentIdx):
                return None
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetWordsFromSentence(word_arr, wordNum, gSentIdx):
                return None
        else:
            return None
        for i in range(wordNum):
            word_list.append(word_arr[i])
    return word_list

def GetPOSsFromSentence(*args):
    """
    restype: the POS list of the sentence (list[str,])
    """
    pos_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        pos_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetPOSsFromSentence_p(pos_arr, wordNum, paraIdx, sentIdx):
                return None
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetPOSsFromSentence(pos_arr, wordNum, gSentIdx):
                return None
        else:
            return None
        for i in range(wordNum):
            pos_list.append(pos_arr[i])
    return pos_list

def GetNEsFromSentence(*args):
    """
    restype: the NE list of the sentence (list[str,])
    """
    ne_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        ne_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetNEsFromSentence_p(ne_arr, wordNum, paraIdx, sentIdx):
                return None
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetNEsFromSentence(ne_arr, wordNum, gSentIdx):
                return None
        else:
            return None
        for i in range(wordNum):
            ne_list.append(ne_arr[i])
    return ne_list

def GetWSDsFromSentence(*args):
    """
    restype: the WSD code list of the sentence (list[str,])
    """
    wsd_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        wsd_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetWSDsFromSentence_p(wsd_arr, wordNum, paraIdx, sentIdx):
                return None
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetWSDsFromSentence(wsd_arr, wordNum, gSentIdx):
                return None
        else:
            return None
        for i in range(wordNum):
            wsd_list.append(wsd_arr[i])
    return wsd_list

def GetWSDExplainsFromSentence(*args):
    """
    restype: the WSD explain list of the sentence (list[str,])
    """
    explain_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        explain_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetWSDExplainsFromSentence_p(explain_arr, wordNum, paraIdx, sentIdx):
                return None
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetWSDExplainsFromSentence(explain_arr, wordNum, gSentIdx):
                return None
        else:
            return None
        for i in range(wordNum):
            explain_list.append(explain_arr[i])
    return explain_list

def GetParsesFromSentence(*args):
    """
    restype: the parent's word idx list and relation type list of the sentence (list[int, ], list[str,])
    """
    parent_list = []
    relate_list = []
    wordNum = CountWordInSentence(*args)
    if wordNum > 0:
        parent_arr = (c_int * wordNum)()
        relate_arr = (c_char_p * wordNum)()
        if len(args) == 2:
            (paraIdx, sentIdx) = args
            if 0 != ltp_dll.GetParsesFromSentence_p(parent_arr, relate_arr, wordNum, paraIdx, sentIdx):
                return (None, None)
        elif len(args) == 1:
            gSentIdx, = args
            if 0 != ltp_dll.GetParsesFromSentence(parent_arr, relate_arr, wordNum, gSentIdx):
                return (None, None)
        else:
            return (None, None)
        for i in range(wordNum):
            parent_list.append(parent_arr[i])
            relate_list.append(relate_arr[i])

    return (parent_list, relate_list)

def CountPredArgToWord(*args):
    """
    restype: the arg num of the word
    argtypes:
        para idx in document, sent idx in para, word idx in sent (int, int, int)   
        or
        global sent idx in document, word idx in sent (int, int)
        or
        global word idx in document (int)        
    """
    if len(args) == 3:
        paraIdx, sentIdx, wordIdx = args
        return ltp_dll.CountPredArgToWord_p_s(paraIdx, sentIdx, wordIdx)
    elif len(args) == 2:
        gSentIdx, wordIdx = args
        return ltp_dll.CountPredArgToWord_s(gSentIdx, wordIdx)
    elif len(args) == 1:
        gWordIdx, = args
        return ltp_dll.CountPredArgToWord(gWordIdx)
    else:
        return -1

def GetPredArgToWord(*args):
    """
    restype: arg type list, arg beg word idx list, arg end word idx list (list[str, ], list[int, ], list[int, ])
    argtypes: as above func
    """
    type_list = []
    beg_list = []
    end_list = []
    wordNum = CountPredArgToWord(*args)
    if (wordNum > 0):
        type_arr = (c_char_p * wordNum)()
        beg_arr = (c_int * wordNum)()
        end_arr = (c_int * wordNum)()
        if len(args) == 3:
            paraIdx, sentIdx, wordIdx = args
            if 0 != ltp_dll.GetPredArgToWord_p_s(type_arr, beg_arr, end_arr, wordNum,
                                                  paraIdx, sentIdx, wordIdx):
                return (None, None, None)
        elif len(args) == 2:
            gSentIdx, wordIdx = args
            if 0 != ltp_dll.GetPredArgToWord_s(type_arr, beg_arr, end_arr, wordNum,
                                               gSentIdx, wordIdx):
                return (None, None, None)
        elif len(args) == 1:
            gWordIdx, = args
            if 0 != ltp_dll.GetPredArgToWord(type_arr, beg_arr, end_arr, wordNum,
                                             gWordIdx):
                return (None, None, None)
        else:
            return (None, None, None)
        for i in range(wordNum):
            type_list.append(type_arr[i])
            beg_list.append(beg_arr[i])
            end_list.append(end_arr[i])

    return (type_list, beg_list, end_list)

