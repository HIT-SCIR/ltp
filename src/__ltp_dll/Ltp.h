#ifndef __LTP_H__
#define __LTP_H__

#include "LTPResource.h"
#include "LTPOption.h"
#include "Xml4nlp.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cassert>

using namespace std;

// extern ofstream ltp_log_file;

class LTP {
public:
    LTP(XML4NLP & xml4nlp);
    LTP(const char * cfg_file, XML4NLP & xml4nlp);
    ~LTP();

    int CreateDOMFromTxt(const char *cszTxtFileName); 
    int CreateDOMFromXml(const char *cszXmlFileName);
    int SaveDOM(const char *cszSaveFileName); 

#ifdef _WIN32 
    int main2(const char *cszFileName, const char *cszSaveFileName, unsigned int flag);
    int splitSentence();
#endif

    int wordseg();
    int postag();
    int ner();
    int parser();
    int srl();

private:
    int splitSentence_dummy();
    int ReadConfFile(const char *confFileName = "ltp_all_modules.conf");

private:
    LTPResource m_ltpResource;
    LTPOption   m_ltpOption;
    XML4NLP &   m_xml4nlp;

    static const unsigned int DO_XML;
    static const unsigned int DO_SPLITSENTENCE;
    static const unsigned int DO_IRLAS;
    static const unsigned int DO_NER;
    static const unsigned int DO_PARSER;
    static const unsigned int DO_SRL;
};

#endif  //  end for __LTP_H__
