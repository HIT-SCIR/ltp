#ifndef __CRF_SEG_ENGINE__
#define __CRF_SEG_ENGINE__

#pragma once
#include "SegEngine.h"
#include "CWSTaggerImpl.h"
#include "crfpp.h"
//#include "../__crf++/crfpp.h"
//#include <param.h>
//#include <tagger.h>

LAS_NS_BEG

class CRFSegEngine : public SegEngine
{
    private:
        CRFPP::Param param;
        //CRFPP::TaggerImpl tagger;
        CRFPP::CWSTaggerImpl tagger;

        static void Gbk2Utf8(const std::string from, std::string &to);
        static void Utf82Gbk(const std::string from, std::string &to);

    public:
        CRFSegEngine(const char *model_name);
        ~CRFSegEngine(void);

        bool Segment(const std::string &text, DictBase *pDict, std::vector<std::string> &vecSegResult);
        bool Segment(const char* text, DictBase *pDict, std::vector<std::string> &vecSegResult);
        bool Segment(const char *pszText, std::vector<std::string> &vecSegResult);

        // input is wide string and the output is a vector of positions
        bool Segment(const std::wstring &text, std::vector<std::pair<int, int> > &vecSegPos);

        std::string ToString() { return "CRFSegEngine";}
};

LAS_NS_END

#endif // __CRF_SEG_ENGINE__
