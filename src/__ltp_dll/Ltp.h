#ifndef _LTP_H_
#define _LTP_H_

#include "LTPResource.h"
#include "LTPOption.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cassert>

using namespace std;

extern ofstream ltp_log_file;
class LTP
{
    public:
        LTP(XML4NLP &xml4nlp);
        ~LTP();

        int CreateDOMFromTxt(const char *cszTxtFileName)
        {
            return m_xml4nlp.CreateDOMFromFile( cszTxtFileName );
        }
        int CreateDOMFromXml(const char *cszXmlFileName)
        {
            return m_xml4nlp.LoadXMLFromFile( cszXmlFileName );
        }
        int SaveDOM(const char *cszSaveFileName)
        {
            return m_xml4nlp.SaveDOM(cszSaveFileName);
        }
#ifdef _WIN32 
        int main2(const char *cszFileName, const char *cszSaveFileName, unsigned int flag);
        int splitSentence();
        //int irlas();
        //int segmentWord();
        //int parser();
#endif

        int crfWordSeg();
        int postag();
        int ner();
        //int wsd();
        int gparser();
        int srl();


    private:
        // DO NOT use these func, use the default manner.
        // There are still some problems to resolve.
        //int SetIRLASOption(const IRLASOption &irlasOpt);
        //int SetNEOption(const NEOption &neOpt);
        //int SetSDSOption(const SDSOption &sdsOpt);

        int splitSentence_dummy();
        int ReadConfFile(const char *confFileName = "ltp_all_modules.conf");

    private:
        LTPResource m_ltpResource;
        LTPOption m_ltpOption;
        XML4NLP &m_xml4nlp;

        static const unsigned int DO_XML;
        static const unsigned int DO_SPLITSENTENCE;
        static const unsigned int DO_IRLAS;
        static const unsigned int DO_NER;
        static const unsigned int DO_PARSER;
        //static const unsigned int DO_WSD;
        static const unsigned int DO_SRL;
};


#endif
