#include "Ltp.h"
#include <ctime>
#include <map>
#include <string>

#include "MyLib.h"
#include "Xml4nlp.h"
#include "SplitSentence.h"
#include "segment_dll.h"
#include "postag_dll.h"
#include "parser_dll.h"
#include "NER_DLL.h"
#include "SRL_DLL.h"

#if _WIN32
#pragma warning(disable: 4786 4284)
#pragma comment(lib, "segmentor.lib")
#pragma comment(lib, "postagger.lib")
#pragma comment(lib, "parser.lib")
#pragma comment(lib, "ner.lib")
#pragma comment(lib, "srl.lib")
#endif

#include "logging.hpp"
#include "cfgparser.hpp"

using namespace std;

const unsigned int LTP::DO_XML = 1;
const unsigned int LTP::DO_SPLITSENTENCE = 1 << 1;
const unsigned int LTP::DO_NER = 1 << 3;
const unsigned int LTP::DO_PARSER = 1 << 4;
const unsigned int LTP::DO_SRL = 1 << 6;

// create a platform
LTP::LTP(XML4NLP &xml4nlp) :
    m_ltpResource(),
    m_ltpOption(),
    m_xml4nlp(xml4nlp) {
    ReadConfFile();
}

LTP::LTP(const char * config, XML4NLP & xml4nlp) :
    m_ltpResource(),
    m_ltpOption(),
    m_xml4nlp(xml4nlp) {
    ReadConfFile(config);
}

LTP::~LTP() {
}

int LTP::CreateDOMFromTxt(const char * cszTxtFileName) {
    return m_xml4nlp.CreateDOMFromFile(cszTxtFileName);
}

int LTP::CreateDOMFromXml(const char * cszXmlFileName) {
    return m_xml4nlp.LoadXMLFromFile(cszXmlFileName);
}

int LTP::SaveDOM(const char * cszSaveFileName) {
    return m_xml4nlp.SaveDOM(cszSaveFileName);
}

int LTP::ReadConfFile(const char * config_file) {
    ltp::utility::ConfigParser cfg(config_file);

    if (!cfg) {
        TRACE_LOG("Failed to open config file \"%s\"", config_file);
        return -1;
    }


    m_ltpOption.segmentor_model_path = "";
    m_ltpOption.postagger_model_path = "";
    m_ltpOption.parser_model_path    = "";
    m_ltpOption.ner_data_dir         = "";
    m_ltpOption.srl_data_dir         = "";

    string buffer;

    if (cfg.get("segmentor-model", buffer)) {
        m_ltpOption.segmentor_model_path = buffer;
    } else {
        WARNING_LOG("No \"segmentor-model\" config is found");
    }

    if (cfg.get("postagger-model", buffer)) {
        m_ltpOption.postagger_model_path = buffer;
    } else {
        WARNING_LOG("No \"postagger-model\" config is found");
    }

    if (cfg.get("parser-model", buffer)) {
        m_ltpOption.parser_model_path = buffer;
    } else {
        WARNING_LOG("No \"parser-model\" config is found");
    }

    if (cfg.get("ner-data", buffer)) {
        m_ltpOption.ner_data_dir = buffer;
    } else {
        WARNING_LOG("No \"ner-data\" config is found");
    }

    if (cfg.get("srl-data", buffer)) {
        m_ltpOption.srl_data_dir = buffer;
    } else {
        WARNING_LOG("No \"srl-data\" config is found");
    }

    m_ltpOption.neOpt.isEntity  = 1;
    m_ltpOption.neOpt.isTime    = 1;
    m_ltpOption.neOpt.isNum     = 1;
    // NE conf, ONLY use default
    /*it = mapConf.find("NE_Entity");
    if (it == mapConf.end() || it->second.empty())
        m_ltpOption.neOpt.isEntity = 1;
    else
        m_ltpOption.neOpt.isEntity = atoi( it->second.c_str() );

    it = mapConf.find("NE_Time");
    if (it == mapConf.end() || it->second.empty())
        m_ltpOption.neOpt.isTime = 1;
    else
        m_ltpOption.neOpt.isTime = atoi( it->second.c_str() );

    it = mapConf.find("NE_Num");
    if (it == mapConf.end() || it->second.empty())
        m_ltpOption.neOpt.isNum = 1;
    else
        m_ltpOption.neOpt.isNum = atoi( it->second.c_str() );*/

    return 0;
}

// If you do NOT split sentence explicitly,
// this will be called according to dependencies among modules
int LTP::splitSentence_dummy() {
    if ( m_xml4nlp.QueryNote(NOTE_SENT) ) {
        return 0;
    }

    int paraNum = m_xml4nlp.CountParagraphInDocument();

    if (paraNum == 0) {
        ERROR_LOG("in LTP::splitsent, There is no paragraph in doc,");
        ERROR_LOG("you may have loaded a blank file or have not loaded a file yet.");
        return -1;
    }

    for (int i = 0; i < paraNum; ++i) {
        vector<string> vecSentences;
        string para;
        m_xml4nlp.GetParagraph(i, para);
        if (0 == SplitSentence( para, vecSentences )) {
            ERROR_LOG("in LTP::splitsent, failed to split sentence");
            return -1;
        }
        // dummy
        // vecSentences.push_back(para);
        if (0 != m_xml4nlp.SetSentencesToParagraph(vecSentences, i)) {
            ERROR_LOG("in LTP::splitsent, failed to write sentence to xml");
            return -1;
        }
    }

    m_xml4nlp.SetNote(NOTE_SENT);
    return 0;
}

// integrate word segmentor into LTP
int LTP::wordseg() {
    if (m_xml4nlp.QueryNote(NOTE_WORD)) {
        return 0;
    }

    //
    if (0 != splitSentence_dummy()) {
        ERROR_LOG("in LTP::wordseg, failed to perform split sentence preprocess.");
        return -1;
    }

    if (0 != m_ltpResource.LoadSegmentorResource(m_ltpOption.segmentor_model_path)) {
        ERROR_LOG("in LTP::wordseg, failed to load segmentor resource");
        return -1;
    }

    // get the segmentor pointer
    void * segmentor = m_ltpResource.GetSegmentor();
    if (0 == segmentor) {
        ERROR_LOG("in LTP::wordseg, failed to init a segmentor");
        return -1;
    }

    int stnsNum = m_xml4nlp.CountSentenceInDocument();

    if (0 == stnsNum) {
        ERROR_LOG("in LTP::wordseg, number of sentence equals 0");
        return -1;
    }

    for (int i = 0; i < stnsNum; ++ i) {
        string strStn = m_xml4nlp.GetSentence(i);
        vector<string> vctWords;

        if (0 == segmentor_segment(segmentor, strStn, vctWords)) {
            ERROR_LOG("in LTP::wordseg, failed to perform word segment on \"%s\"",
                    strStn.c_str());
            return -1;
        }

        if (0 != m_xml4nlp.SetWordsToSentence(vctWords, i)) {
            ERROR_LOG("in LTP::wordseg, failed to write segment result to xml");
            return -1;
        }
    }

    m_xml4nlp.SetNote(NOTE_WORD);
    return 0;
}

// integrate postagger into LTP
int LTP::postag() {
    if ( m_xml4nlp.QueryNote(NOTE_POS) ) {
        return 0;
    }

    // dependency
    if (0 != wordseg()) {
        ERROR_LOG("in LTP::postag, failed to perform word segment preprocess");
        return -1;
    }

    if (0 != m_ltpResource.LoadPostaggerResource(m_ltpOption.postagger_model_path)) {
        ERROR_LOG("in LTP::postag, failed to load postagger resource.");
        return -1;
    }

    void * postagger = m_ltpResource.GetPostagger();
    if (0 == postagger) {
        ERROR_LOG("in LTP::postag, failed to init a postagger");
        return -1;
    }

    int stnsNum = m_xml4nlp.CountSentenceInDocument();

    if (0 == stnsNum) {
        ERROR_LOG("in LTP::postag, number of sentence equals 0");
        return -1;
    }

    for (int i = 0; i < stnsNum; ++i) {
        vector<string> vecWord;
        vector<string> vecPOS;

        m_xml4nlp.GetWordsFromSentence(vecWord, i);
        if (0 == postagger_postag(postagger, vecWord, vecPOS)) {
            ERROR_LOG("in LTP::postag, failed to perform postag on sent. #%d", i+1);
            return -1;
        }

        if (m_xml4nlp.SetPOSsToSentence(vecPOS, i) != 0) {
            ERROR_LOG("in LTP::postag, failed to write postag result to xml");
            return -1;
        }
    }

    m_xml4nlp.SetNote(NOTE_POS);

    return 0;
}

// perform ner over xml
int LTP::ner() {
    if ( m_xml4nlp.QueryNote(NOTE_NE) ) {
        return 0;
    }

    // dependency
    if (0 != postag()) {
        ERROR_LOG("in LTP::ner, failed to perform postag preprocess");
        return -1;
    }

    if (0 != m_ltpResource.LoadNEResource(m_ltpOption.ner_data_dir)) {
        ERROR_LOG("in LTP::ner, failed to load ner resource");
        return -1;
    }

    void * ner = m_ltpResource.GetNER();

    if (NULL == ner) {
        ERROR_LOG("in LTP::ner, failed to init a ner.");
        return -1;
    }

    NER_SetOption(m_ltpOption.neOpt.isEntity,
            m_ltpOption.neOpt.isTime,
            m_ltpOption.neOpt.isNum);

    int stnsNum = m_xml4nlp.CountSentenceInDocument();

    if (stnsNum == 0) {
        ERROR_LOG("in LTP::ner, number of sentence equals 0");
        return -1;
    }

    for (int i = 0; i < stnsNum; ++ i) {
        vector<string> vecWord;
        vector<string> vecPOS;
        vector<string> vecNETag;

        if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get words from xml");
            return -1;
        }

        if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get postags from xml");
            return -1;
        }

        if (0 != NER(ner, vecWord, vecPOS, vecNETag)) {
            ERROR_LOG("in LTP::ner, failed to perform ner on sent. #%d", i+1);
            return -1;
        }

        m_xml4nlp.SetNEsToSentence(vecNETag, i);
    }

    m_xml4nlp.SetNote(NOTE_NE);
    return 0;
}

int LTP::parser() {
    if ( m_xml4nlp.QueryNote(NOTE_PARSER) ) return 0;

    if (0 != postag()) {
        ERROR_LOG("in LTP::parser, failed to perform postag preprocessing");
        return -1;
    }

    if ( 0 != m_ltpResource.LoadParserResource(m_ltpOption.parser_model_path) ) {
        ERROR_LOG("in LTP::parser, failed to load parser resource");
        return -1;
    }

    void * parser = m_ltpResource.GetParser();

    if (parser == NULL) {
        ERROR_LOG("in LTP::parser, failed to init a parser");
        return -1;
    }

    int stnsNum = m_xml4nlp.CountSentenceInDocument();
    if (stnsNum == 0) {
        ERROR_LOG("in LTP::parser, number of sentences equals 0");
        return -1;
    }

    for (int i = 0; i < stnsNum; ++i) {
        vector<string>  vecWord;
        vector<string>  vecPOS;
        vector<int>     vecHead;
        vector<string>  vecRel;

        if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) {
            ERROR_LOG("in LTP::parser, failed to get words from xml");
            return -1;
        }

        if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) {
            ERROR_LOG("in LTP::parser, failed to get postags from xml");
            return -1;
        }

        if (-1 == parser_parse(parser, vecWord, vecPOS, vecHead, vecRel)) {
            ERROR_LOG("in LTP::parser, failed to perform parse on sent. #%d", i+1);
            return -1;
        }

        if (0 != m_xml4nlp.SetParsesToSentence(vecHead, vecRel, i)) {
            ERROR_LOG("in LTP::parser, failed to write parse result to xml");
            return -1;
        }
    }

    m_xml4nlp.SetNote(NOTE_PARSER);

    return 0;
}

int LTP::srl() {
    if ( m_xml4nlp.QueryNote(NOTE_SRL) ) return 0;

    // dependency
    if (0 != ner()) {
        ERROR_LOG("in LTP::srl, failed to perform ner preprocess");
        return -1;
    }

    if (0 != parser()) {
        ERROR_LOG("in LTP::srl, failed to perform parsing preprocess");
        return -1;
    }

    if ( 0 != m_ltpResource.LoadSRLResource(m_ltpOption.srl_data_dir) ) {
        ERROR_LOG("in LTP::srl, failed to load srl resource");
        return -1;
    }

    int stnsNum = m_xml4nlp.CountSentenceInDocument();
    if (stnsNum == 0) {
        ERROR_LOG("in LTP::srl, number of sentence equals 0");
        return -1;
    }

    for (int i = 0; i < stnsNum; ++i) {
        vector<string>              vecWord;
        vector<string>              vecPOS;
        vector<string>              vecNE;
        vector< pair<int, string> > vecParse;
        vector< pair< int, vector< pair<const char *, pair< int, int > > > > > vecSRLResult;

        if (m_xml4nlp.GetWordsFromSentence(vecWord, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get words from xml");
            return -1;
        }

        if (m_xml4nlp.GetPOSsFromSentence(vecPOS, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get postags from xml");
            return -1;
        }

        if (m_xml4nlp.GetNEsFromSentence(vecNE, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get ner result from xml");
            return -1;
        }

        if (m_xml4nlp.GetParsesFromSentence(vecParse, i) != 0) {
            ERROR_LOG("in LTP::ner, failed to get parsing result from xml");
            return -1;
        }

        if (0 != SRL(vecWord, vecPOS, vecNE, vecParse, vecSRLResult)) {
            ERROR_LOG("in LTP::srl, failed to perform srl on sent. #%d", i+1);
            return -1;
        }

        int j = 0;
        for (; j < vecSRLResult.size(); ++j) {
            vector<string>              vecType;
            vector< pair<int, int> >    vecBegEnd;
            int k = 0;

            for (; k < vecSRLResult[j].second.size(); ++k) {
                vecType.push_back(vecSRLResult[j].second[k].first);
                vecBegEnd.push_back(vecSRLResult[j].second[k].second);
            }

            if (0 != m_xml4nlp.SetPredArgToWord(i, vecSRLResult[j].first, vecType, vecBegEnd)) {
                return -1;
            }
        }
    }

    m_xml4nlp.SetNote(NOTE_SRL);
    return 0;
}

