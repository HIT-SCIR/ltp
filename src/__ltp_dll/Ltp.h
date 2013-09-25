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
#define MAX_SENTENCE_LEN 300
#define MAX_WORDS_NUM    70

class LTP {
public:
    LTP();
    LTP(const char * cfg_file);
    ~LTP();

    int wordseg(XML4NLP & xml4nlp);
    int postag(XML4NLP & xml4nlp);
    int ner(XML4NLP & xml4nlp);
    int parser(XML4NLP & xml4nlp);
    int srl(XML4NLP & xml4nlp);

private:
    int splitSentence_dummy(XML4NLP & xml4nlp);
    int ReadConfFile(const char *confFileName = "conf/ltp.cnf");

private:
    LTPResource m_ltpResource;
    LTPOption   m_ltpOption;

    static const unsigned int DO_XML;
    static const unsigned int DO_SPLITSENTENCE;
    static const unsigned int DO_IRLAS;
    static const unsigned int DO_NER;
    static const unsigned int DO_PARSER;
    static const unsigned int DO_SRL;
};

#endif  //  end for __LTP_H__
