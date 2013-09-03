/*
 * HIT-IRLab (c) 2001-2005, all rights reserved.
 * This software is "XML Text Representation for NLP"
 * Its aim is to integrate all the modules of IRLab into a uniform frame
 * The author of this software if Huipeng Zhang (zhp@ir.hit.edu.cn)
 * The create time of this software is 2005-11-01
 * In this software, a open source XML parser TinyXML is used
 * We Thank to the author of it -- Lee Thomason
 */

#include "Xml4nlp.h"
#include "MyLib.h"

const char * const NOTE_SENT        = "sent";
const char * const NOTE_WORD        = "word";
const char * const NOTE_POS         = "pos";
const char * const NOTE_NE          = "ne";
const char * const NOTE_PARSER      = "parser";
const char * const NOTE_WSD         = "wsd";
const char * const NOTE_SRL         = "srl";
//const char * const NOTE_CLASS = "class";
//const char * const NOTE_SUM = "sum";
//const char * const NOTE_CR = "cr";

const char * const XML4NLP::TAG_DOC         = "doc";
const char * const XML4NLP::TAG_NOTE        = "note";
const char * const XML4NLP::TAG_SUM         = "sum";
const char * const XML4NLP::TAG_CLASS       = "class";
const char * const XML4NLP::TAG_COREF       = "coref";
const char * const XML4NLP::TAG_COREF_CR    = "cr";
const char * const XML4NLP::TAG_COREF_MENT  = "mention";
const char * const XML4NLP::TAG_PARA        = "para";
const char * const XML4NLP::TAG_SENT        = "sent";
const char * const XML4NLP::TAG_WORD        = "word";
const char * const XML4NLP::TAG_CONT        = "cont";
const char * const XML4NLP::TAG_POS         = "pos";
const char * const XML4NLP::TAG_NE          = "ne";
const char * const XML4NLP::TAG_PSR_PARENT  = "parent";
const char * const XML4NLP::TAG_PSR_RELATE  = "relate";
const char * const XML4NLP::TAG_WSD         = "wsd";
const char * const XML4NLP::TAG_WSD_EXP     = "wsdexp";
const char * const XML4NLP::TAG_SRL_ARG     = "arg";
const char * const XML4NLP::TAG_SRL_TYPE    = "type";
const char * const XML4NLP::TAG_BEGIN       = "beg";
const char * const XML4NLP::TAG_END         = "end";
const char * const XML4NLP::TAG_ID          = "id";

XML4NLP::XML4NLP() {
    m_document.documentPtr = NULL; 
    m_note.nodePtr           = NULL;
    m_summary.nodePtr        = NULL;
    m_textclass.nodePtr      = NULL;
    m_coref.nodePtr          = NULL;
}

XML4NLP::~XML4NLP() {
    m_tiXmlDoc.Clear();
}

/////////////////////////////////////////////////////////////////////////////////////
/// read a raw text file and create a initial DOM tree.
/// the paragraphs are separated by CR ("\r\n")
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::CreateDOMFromFile(const char* fileName) {
    ClearDOM();

    if (0 != BuildDOMFrame()) return -1;

    ifstream in;
    in.open(fileName);
    if ( !in.is_open() ) {
        cerr << "xml4nlp load file error: " << fileName << endl;
        return -1;
    }

    string line;
    int i = 0;
    while (getline(in, line)) {
        clean_str(line); // Zhenghua Li, 2007-8-31, 15:57
        // remove_space_gbk(line);
        if (line.empty()) {
            continue;
        }

        if (0 != BuildParagraph(line, i++)) return -1;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// read raw text from a string and create a initial DOM tree.
/// the paragraphs are separated by CR ("\r\n")
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::CreateDOMFromString(const string & str) {
    ClearDOM();

    if (0 != BuildDOMFrame()) return -1;

    string strTmp = str;
    replace_char_by_char(strTmp, '\r', '\n');

    // std::cout << strTmp << std::endl;
    istringstream in(strTmp);	// How to use istringstream?
    string line;
    int i = 0;
    while (getline(in, strTmp)) {
        clean_str(strTmp);

        if (strTmp.empty()) {
            continue;
        }

        if (0 != BuildParagraph(strTmp, i++)) {
            return -1;
        }
    }

    return 0;
}

void XML4NLP::ReportTiXmlDocErr() const {
    cerr << "[XML4NLP ERROR REPORT]" << endl;
    cerr << "description : " << m_tiXmlDoc.ErrorDesc() << endl;
    cerr << "location :    " << endl;
    cerr << "row :         " << m_tiXmlDoc.ErrorRow() << endl;
    cerr << "col :         " << m_tiXmlDoc.ErrorCol() << endl;
    cerr << "=====================" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////
/// load a xml file and parse it.
/// it includes two phases:
/// 1. call LoadFile() which construct a DOM tree
/// 2. initialize private members of the class Document, Paragraph and Sentence
/// note: the input file must be a XML file.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::LoadXMLFromFile(const char* fileName) {
    ClearDOM();

    if ( !m_tiXmlDoc.LoadFile(fileName) ) {
        cerr << "load xml file error: " << fileName << endl;
        ReportTiXmlDocErr();
        return -1;
    }

    return InitXmlStructure();
}

/////////////////////////////////////////////////////////////////////////////////////
/// load a xml file from a string and parse it.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::LoadXMLFromString(const char * str) {
    ClearDOM();
    m_tiXmlDoc.Parse(str);

    if (m_tiXmlDoc.Error()) {
        ReportTiXmlDocErr();
        return -1;
    }

    if (-1 == InitXmlStructure()) {
        return -1;
    }

    if (!LTMLValidation()) {
        // failed LTML Validation
        return -1;
    }

    return 0;
}

int XML4NLP::LoadXMLFromString(const std::string & str) {
    LoadXMLFromString(str.c_str());
}

/////////////////////////////////////////////////////////////////////////////////////
/// clear the DOM tree, delete all nodes that allocated before.
/////////////////////////////////////////////////////////////////////////////////////
void XML4NLP::ClearDOM() {
    m_tiXmlDoc.Clear();

    m_document.documentPtr = NULL;
    m_document.paragraphs.clear();
    m_note.nodePtr = NULL;
    m_summary.nodePtr = NULL;
    m_textclass.nodePtr = NULL;
    m_coref.nodePtr = NULL;
    m_coref.vecEntity.clear();

    m_vecBegWordIdxOfStns.clear();
    m_vecBegStnsIdxOfPara.clear();
}

/////////////////////////////////////////////////////////////////////////////////////
/// save the DOM tree to a XML file.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::SaveDOM(const char* fileName) {
    if (!m_tiXmlDoc.SaveFile(fileName)) {
        ReportTiXmlDocErr();
        return -1;
    }

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// save the DOM tree to a XML string.
/////////////////////////////////////////////////////////////////////////////////////
void XML4NLP::SaveDOM(string &strDocument) const {
    TiXmlPrinter printer;
    m_tiXmlDoc.Accept(&printer);
    strDocument = printer.CStr();
}

// ----------------------------------------------------------------some counting functions
int XML4NLP::CountParagraphInDocument() const {
    return m_document.paragraphs.size();
}

int XML4NLP::CountSentenceInParagraph(int paragraphIdx) const {
    if ( 0 != CheckRange(paragraphIdx) ) return 0;
    return m_document.paragraphs[paragraphIdx].sentences.size();
}

int XML4NLP::CountSentenceInDocument() const {
    int stnsNumInDoc = 0;
    int paragraphNum = m_document.paragraphs.size();
    for (int i = 0; i < paragraphNum; ++i) {
        stnsNumInDoc += m_document.paragraphs[i].sentences.size();
    }
    return stnsNumInDoc;
}

int XML4NLP::CountWordInSentence(int paragraphIdx, int sentenceIdx) const {
    if ( 0 != CheckRange(paragraphIdx, sentenceIdx) ) return 0;
    return m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words.size();
}

int XML4NLP::CountWordInSentence(int sentenceIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if ( 0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx) ) return 0;
    return m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words.size();
}

int XML4NLP::CountWordInParagraph(int paragraphIdx) const {
    if ( 0 != CheckRange(paragraphIdx) ) return -1;
    int totalWordNum = 0;
    int sentNum = m_document.paragraphs[paragraphIdx].sentences.size();
    for (int i=0; i < sentNum; ++i) {
        totalWordNum += m_document.paragraphs[paragraphIdx].sentences[i].words.size();
    }
    return totalWordNum;
}

int XML4NLP::CountWordInDocument() const {
    int totalWordNum = 0;
    int paraNum = m_document.paragraphs.size();
    for (int i=0; i<paraNum; ++i) {
        int sentNum = m_document.paragraphs[i].sentences.size();
        for (int j=0; j<sentNum; ++j) {
            totalWordNum += m_document.paragraphs[i].sentences[j].words.size();
        }
    }
    return totalWordNum;
}

const char * XML4NLP::GetParagraph(int paragraphIdx) const {
    if (0 != CheckRange(paragraphIdx)) return NULL;
    if (QueryNote(NOTE_SENT)) return NULL;

    TiXmlElement *paraPtr = m_document.paragraphs[paragraphIdx].paragraphPtr;
    return paraPtr->GetText();
}

int XML4NLP::GetParagraph(int paragraphIdx, string &strParagraph) const {
    if (0 != CheckRange(paragraphIdx)) {
        return -1;
    }

    const Paragraph_t &paragraph = m_document.paragraphs[paragraphIdx];

    if (paragraph.sentences.empty()) {
        strParagraph = paragraph.paragraphPtr->GetText() ;
    } else {
        strParagraph = "";
        const vector<Sentence_t> &sentences = paragraph.sentences;
        for (int i=0; i<sentences.size(); ++i) {
            strParagraph += sentences[i].sentencePtr->Attribute(TAG_CONT);
        }
    }

    return 0;
}

const char* XML4NLP::GetSentence(int paragraphIdx, int sentenceIdx) const {
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return NULL;
    return m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].sentencePtr->Attribute(TAG_CONT);
}

const char* XML4NLP::GetSentence(int sentenceIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return NULL;
    return GetSentence(paraIdx_sentIdx.first, paraIdx_sentIdx.second);
}

const char* XML4NLP::GetWord(int paragraphIdx, int sentenceIdx, int wordIdx) const {
    if ( 0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx) ) return NULL;
    return m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_CONT);
}

const char* XML4NLP::GetWord(int globalSentIdx, int wordIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return NULL;
    return GetWord(paraIdx_sentIdx.first, paraIdx_sentIdx.second, wordIdx);
}

const char* XML4NLP::GetWord(int globalWordIdx) const {
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return NULL;
    return GetWord(paraIdx, sentIdx, wordIdx);
}

const char *XML4NLP::GetPOS(int paragraphIdx, int sentenceIdx, int wordIdx) const {
    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return NULL;
    return m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_POS);
}

const char *XML4NLP::GetPOS(int globalSentIdx, int wordIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return NULL;
    return GetPOS(paraIdx_sentIdx.first, paraIdx_sentIdx.second, wordIdx);
}

const char *XML4NLP::GetPOS(int globalWordIdx) const {
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return NULL;
    return GetPOS(paraIdx, sentIdx, wordIdx);
}

const char *XML4NLP::GetNE(int paragraphIdx, int sentenceIdx, int wordIdx) const {
    if ( 0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx) ) return NULL;
    return m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_NE);
}

const char *XML4NLP::GetNE(int globalSentIdx, int wordIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return NULL;
    return GetNE(paraIdx_sentIdx.first, paraIdx_sentIdx.second, wordIdx);
}

const char *XML4NLP::GetNE(int globalWordIdx) const {
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return NULL;
    return GetNE(paraIdx, sentIdx, wordIdx);
}


int XML4NLP::GetWSD(pair<const char *, const char *> &WSD_explanation, 
        int paragraphIdx, 
        int sentenceIdx, 
        int wordIdx) const {
   if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;

   WSD_explanation.first  = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_WSD);
   WSD_explanation.second = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_WSD_EXP);
   return 0;
}

int XML4NLP::GetWSD(pair<const char *, const char *> & WSD_explanation, 
        int globalSentIdx, 
        int wordIdx) const {
   pair<int, int> paraIdx_sentIdx;
   if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return -1;
   return GetWSD(WSD_explanation, paraIdx_sentIdx.first, paraIdx_sentIdx.second);
}

int XML4NLP::GetWSD(pair<const char *, const char *> & WSD_explanation, 
        int globalWordIdx) const {
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return -1;
    return GetWSD(WSD_explanation, paraIdx, sentIdx, wordIdx);
}

int XML4NLP::GetParse(pair<int, const char *> & parent_relation, 
        int paragraphIdx, 
        int sentenceIdx, 
        int wordIdx) const {
    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;
    const char *cszParent = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_PSR_PARENT);
    parent_relation.first   = (cszParent == NULL ? 0 : atoi(cszParent));
    parent_relation.second  = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr->Attribute(TAG_PSR_RELATE);
    return 0;
}

int XML4NLP::GetParse(pair<int, const char *> & parent_relation, 
        int globalSentIdx, 
        int wordIdx) const {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return -1;
    return GetParse(parent_relation, paraIdx_sentIdx.first, paraIdx_sentIdx.second);
}

int XML4NLP::GetParse(pair<int, const char *> &parent_relation, 
        int globalWordIdx) const {
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return -1;
    return GetParse(parent_relation, paraIdx, sentIdx, wordIdx);
}

int XML4NLP::MapGlobalSentIdx2paraIdx_sentIdx(int sentenceIdx, 
        pair<int, int> & paraIdx_sentIdx) const {
    int startStnsIdxOfPara = 0;
    for (int paraIdx=0; paraIdx < m_document.paragraphs.size(); ++paraIdx) {
        if (startStnsIdxOfPara + m_document.paragraphs[paraIdx].sentences.size() > sentenceIdx) {
            paraIdx_sentIdx.first = paraIdx;
            paraIdx_sentIdx.second = sentenceIdx - startStnsIdxOfPara;
            return 0;
        }
        startStnsIdxOfPara += m_document.paragraphs[paraIdx].sentences.size();
    }
    return -1;
}

int XML4NLP::MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(int globalWordIdx, 
        int & paraIdx, 
        int & sentIdx, 
        int & wordIdx) const {
    int startWordIdxOfStns = 0;
    for (paraIdx=0; paraIdx < m_document.paragraphs.size(); ++paraIdx) {
        const vector<Sentence_t> &sentences = m_document.paragraphs[paraIdx].sentences;
        for (sentIdx=0; sentIdx < sentences.size(); ++sentIdx) {
            if (startWordIdxOfStns + sentences[sentIdx].words.size() > globalWordIdx) {
                wordIdx = globalWordIdx - startWordIdxOfStns;
                return 0;
            }
            startWordIdxOfStns += sentences[sentIdx].words.size();
        }
    }
    return -1;
}

int XML4NLP::GetSentencesFromParagraph(vector<const char *> &vecSentence, 
        int paragraphIdx) const {
    if (0 != CheckRange(paragraphIdx)) return -1;
    if (m_document.paragraphs[paragraphIdx].sentences.empty()) {
        return -1;
    }

    const vector<Sentence_t> & sentences = m_document.paragraphs[paragraphIdx].sentences;
    if (vecSentence.size() != sentences.size()) {
        return -1;
    }

    for (int i=0; i < sentences.size(); ++i) {
        vecSentence[i] = sentences[i].sentencePtr->Attribute(TAG_CONT);
    }

    return 0;
}

int XML4NLP::GetSentencesFromParagraph(vector<string> &vecSentence, 
        int paragraphIdx) const {
    if (0 != CheckRange(paragraphIdx)) return -1;

    if (m_document.paragraphs[paragraphIdx].sentences.empty()) {
        return -1;
    }

    vecSentence.clear();
    const vector<Sentence_t> &sentences = m_document.paragraphs[paragraphIdx].sentences;
    for (int i = 0; i < sentences.size(); ++ i) {
        vecSentence.push_back( sentences[i].sentencePtr->Attribute(TAG_CONT) );
    }
    return 0;
}

int XML4NLP::SetSentencesToParagraph(const vector<string> &vecSentence, int paragraphIdx) {
    if (0 != CheckRange(paragraphIdx)) {
        return -1;
    }

    if (!m_document.paragraphs[paragraphIdx].sentences.empty()) {
        return -1;
    }

    Paragraph_t & paragraph         = m_document.paragraphs[paragraphIdx];
    TiXmlElement * paragraphPtr     = paragraph.paragraphPtr;
    vector<Sentence_t> &sentences   = paragraph.sentences;

    TiXmlText *textPtr = paragraphPtr->FirstChild()->ToText();
    if (textPtr == NULL) {
        return -1;
    } else {
        paragraphPtr->RemoveChild(textPtr);
    }

    for (int i = 0; i < vecSentence.size(); ++i) {
        TiXmlElement *sentencePtr = new TiXmlElement(TAG_SENT);
        sentencePtr->SetAttribute(TAG_ID, static_cast<int>(i));
        sentencePtr->SetAttribute(TAG_CONT, vecSentence[i].c_str());
        paragraphPtr->LinkEndChild(sentencePtr);

        sentences.push_back( Sentence_t() );
        sentences[sentences.size()-1].sentencePtr = sentencePtr;
    }

    return 0;
}

int XML4NLP::GetWordsFromSentence(vector<const char *> & vecWord,
        int paragraphIdx,
        int sentenceIdx) const {
    return GetInfoFromSentence(vecWord, paragraphIdx, sentenceIdx, TAG_CONT);
}

int XML4NLP::GetWordsFromSentence(std::vector<std::string> & vecWord,
        int paragraphIdx,
        int sentenceIdx) const {
    return GetInfoFromSentence(vecWord, paragraphIdx, sentenceIdx, TAG_CONT);
}

int XML4NLP::GetWordsFromSentence(std::vector<const char *> & vecWord,
        int globalSentIdx) const {
    return GetInfoFromSentence(vecWord, globalSentIdx, TAG_CONT);
}

int XML4NLP::GetWordsFromSentence(std::vector<std::string> & vecWord,
        int globalSentIdx) const {
    return GetInfoFromSentence(vecWord, globalSentIdx, TAG_CONT);
}

int XML4NLP::SetWordsToSentence(const vector<string> & vecWord, 
        int paragraphIdx, 
        int sentenceIdx) {
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;

    Sentence_t &sentence = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx];
    if (!sentence.words.empty()) {
        return -1;
    }

    for (int i = 0; i < vecWord.size(); ++ i) {
        TiXmlElement *wordPtr = new TiXmlElement(TAG_WORD);
        wordPtr->SetAttribute(TAG_ID, i);
        wordPtr->SetAttribute(TAG_CONT, vecWord[i].c_str());
        sentence.sentencePtr->LinkEndChild(wordPtr);

        sentence.words.push_back( Word_t() );
        sentence.words[sentence.words.size() - 1].wordPtr = wordPtr;
    }
    return 0;
}

int XML4NLP::SetWordsToSentence(const vector<string> & vecWord,
        int sentenceIdx) {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) {
        return -1;
    }
    SetWordsToSentence(vecWord, paraIdx_sentIdx.first, paraIdx_sentIdx.second);
    return 0;
}

int XML4NLP::GetPOSsFromSentence(std::vector<const char *> & vecPOS,
        int paragraphIdx,
        int sentenceIdx) const {
    return GetInfoFromSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
}

int XML4NLP::GetPOSsFromSentence(std::vector<const char *> &vecPOS, 
        int globalSentIdx) const {
    return GetInfoFromSentence(vecPOS, globalSentIdx, TAG_POS);
}

int XML4NLP::GetPOSsFromSentence(std::vector<std::string> &vecPOS, 
        int paragraphIdx, 
        int sentenceIdx) const {
    return GetInfoFromSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
}

int XML4NLP::GetPOSsFromSentence(std::vector<std::string> &vecPOS, 
        int globalSentIdx) const {
    return GetInfoFromSentence(vecPOS, globalSentIdx, TAG_POS);
}

int XML4NLP::SetPOSsToSentence(const std::vector<std::string> &vecPOS,
        int paragraphIdx,
        int sentenceIdx) {
    return SetInfoToSentence(vecPOS, paragraphIdx, sentenceIdx, TAG_POS);
}

int XML4NLP::SetPOSsToSentence(const std::vector<std::string> & vecPOS,
        int sentenceIdx) {
    return SetInfoToSentence(vecPOS, sentenceIdx, TAG_POS);
}

int XML4NLP::GetNEsFromSentence(std::vector<const char *> & vecNE,
        int paragraphIdx,
        int sentenceIdx) const {
    return GetInfoFromSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
}

int XML4NLP::GetNEsFromSentence(std::vector<const char *> & vecNE, 
        int globalSentIdx) const {
    return GetInfoFromSentence(vecNE, globalSentIdx, TAG_NE);
}

int XML4NLP::GetNEsFromSentence(std::vector<std::string> &vecNE, 
        int paragraphIdx, 
        int sentenceIdx) const {
    return GetInfoFromSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
}

int XML4NLP::GetNEsFromSentence(std::vector<std::string> & vecNE, 
        int globalSentIdx) const {
    return GetInfoFromSentence(vecNE, globalSentIdx, TAG_NE);
}

int XML4NLP::SetNEsToSentence(const std::vector<std::string> & vecNE, 
        int paragraphIdx, 
        int sentenceIdx) {
    return SetInfoToSentence(vecNE, paragraphIdx, sentenceIdx, TAG_NE);
}

int XML4NLP::SetNEsToSentence(const std::vector<std::string> & vecNE,
        int sentenceIdx) {
    return SetInfoToSentence(vecNE, sentenceIdx, TAG_NE);
}

int XML4NLP::GetParsesFromSentence(vector< pair<int, const char *> > &vecParse, 
        int paragraphIdx, 
        int sentenceIdx) const {
    vector<const char *> vecParent;
    vector<const char *> vecRelate;
    int wordNum = CountWordInSentence(paragraphIdx, sentenceIdx);
    if (wordNum != vecParse.size()) {
        cerr << "vecParse.size() does not equal to the word num in the sentence, should resize first" << endl;
        return -1;
    }

    // vecParent.resize(wordNum);
    // vecRelate.resize(wordNum);
    if (0 != GetInfoFromSentence(vecParent, paragraphIdx, sentenceIdx, TAG_PSR_PARENT)) {
        return -1;
    }

    if (0 != GetInfoFromSentence(vecRelate, paragraphIdx, sentenceIdx, TAG_PSR_RELATE)) {
        return -1;
    }

    for (int i=0; i < vecParent.size(); ++ i) {
        vecParse[i].first = atoi( vecParent[i] );
        vecParse[i].second = vecRelate[i];
    }

    return 0;
}

int XML4NLP::GetParsesFromSentence(vector< pair<int, const char *> > & vecParse, 
        int sentenceIdx) const {
    vector<const char *> vecParent;
    vector<const char *> vecRelate;
    int wordNum = CountWordInSentence(sentenceIdx);
    if (wordNum != vecParse.size()) {
        cerr << "vecParse.size() does not equal to the word num in the sentence, should resize first" << endl;
        return -1;
    }

    vecParent.resize(wordNum);
    vecRelate.resize(wordNum);
    if (0 != GetInfoFromSentence(vecParent, sentenceIdx, TAG_PSR_PARENT)) {
        return -1;
    }

    if (0 != GetInfoFromSentence(vecRelate, sentenceIdx, TAG_PSR_RELATE)) {
        return -1;
    }

    for (int i=0; i < vecParent.size(); ++i) {
        vecParse[i].first = atoi( vecParent[i] );
        vecParse[i].second = vecRelate[i];
    }

    return 0;
}


int XML4NLP::GetParsesFromSentence(vector< pair<int, string> > &vecParse, 
        int paragraphIdx, 
        int sentenceIdx) const {
    vector<string> vecParent;
    vector<string> vecRelate;
    if (0 != GetInfoFromSentence(vecParent, paragraphIdx, sentenceIdx, TAG_PSR_PARENT)) {
        return -1;
    }
    if (0 != GetInfoFromSentence(vecRelate, paragraphIdx, sentenceIdx, TAG_PSR_RELATE)) {
        return -1;
    }

    vecParse.clear();
    // Assume their sizes of the two vector are equal. Is it OK?
    for (int i=0; i < vecParent.size(); ++i) {
        int parentIdx = atoi( vecParent[i].c_str() );
        vecParse.push_back( make_pair(static_cast<int>(parentIdx), vecRelate[i]) );
    }

    return 0;
}

int XML4NLP::GetParsesFromSentence(vector< pair<int, string> > &vecParse, 
        int sentenceIdx) const {
    vector<string> vecParent;
    vector<string> vecRelate;
    if (0 != GetInfoFromSentence(vecParent, sentenceIdx, TAG_PSR_PARENT)) {
        return -1;
    }
    if (0 != GetInfoFromSentence(vecRelate, sentenceIdx, TAG_PSR_RELATE)) {
        return -1;
    }

    vecParse.clear();
    // Assume their sizes of the two vector are equal. Is it OK?
    for (int i=0; i < vecParent.size(); ++i) {
        int parentIdx = atoi( vecParent[i].c_str() );
        vecParse.push_back( make_pair(static_cast<int>(parentIdx), vecRelate[i]) );
    }

    return 0;
}

int XML4NLP::SetParsesToSentence(const vector< pair<int, string> > &vecParse, 
        int paragraphIdx, 
        int sentenceIdx) {
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;
    vector<Word_t> &words = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words;
    if (words.size() != vecParse.size()) {
        cerr << "word number does not equal to vecInfo's size in paragraph" << paragraphIdx
            << " sentence " << sentenceIdx << endl;
        return -1;
    }

    if (words[0].wordPtr->Attribute(TAG_PSR_PARENT) != NULL) {
        cerr << "Attribute \"" << TAG_PSR_PARENT << "\" already exists in paragraph" << paragraphIdx
            << " sentence " << sentenceIdx << endl;
        return -1;
    }

    if (words[0].wordPtr->Attribute(TAG_PSR_RELATE) != NULL) {
        cerr << "Attribute \"" << TAG_PSR_RELATE << "\" already exists in paragraph" << paragraphIdx
            << " sentence " << sentenceIdx << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i) {
        words[i].wordPtr->SetAttribute(TAG_PSR_PARENT, vecParse[i].first);
        words[i].wordPtr->SetAttribute(TAG_PSR_RELATE, vecParse[i].second.c_str());
    }

    return 0;
}

int XML4NLP::SetParsesToSentence(const vector< pair<int, string> > &vecParse, int sentenceIdx) {
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    vector<Word_t> &words = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words;
    if (words.size() != vecParse.size()) {
        cerr << "word number does not equal to vecInfo's size in paragraph" << paraIdx_sentIdx.first
            << " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(TAG_PSR_PARENT) != NULL)
    {
        cerr << "Attribute \"" << TAG_PSR_PARENT << "\" already exists in paragraph" << paraIdx_sentIdx.first
            << " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(TAG_PSR_RELATE) != NULL)
    {
        cerr << "Attribute \"" << TAG_PSR_RELATE << "\" already exists in paragraph" << paraIdx_sentIdx.first
            << " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        words[i].wordPtr->SetAttribute(TAG_PSR_PARENT, vecParse[i].first);
        words[i].wordPtr->SetAttribute(TAG_PSR_RELATE, vecParse[i].second.c_str());
    }
    return 0;
}

int XML4NLP::SetParsesToSentence(const vector<int> &vecHead, const vector<string> &vecRel, int paragraphIdx, int sentenceIdx)
{
    if (0 != SetInfoToSentence(vecHead, paragraphIdx, sentenceIdx, TAG_PSR_PARENT)) return -1;
    if (0 != SetInfoToSentence(vecRel, paragraphIdx, sentenceIdx, TAG_PSR_RELATE)) return -1;
    return 0;
}

int XML4NLP::SetParsesToSentence(const vector<int> &vecHead, const vector<string> &vecRel, int sentenceIdx)
{
    // decreasing vecHead index
    vector<int> d_vecHead;
    for (int i = 0; i < vecHead.size(); i++)
    {
        d_vecHead.push_back(vecHead[i] - 1);
    }

    if (0 != SetInfoToSentence(d_vecHead, sentenceIdx, TAG_PSR_PARENT)) return -1;
    if (0 != SetInfoToSentence(vecRel, sentenceIdx, TAG_PSR_RELATE)) return -1;
    return 0;
}

// ----------------------------------------------------------------for text summarization
const char* XML4NLP::GetTextSummary() const
{
    if (m_summary.nodePtr != NULL)
    {
        return m_summary.nodePtr->GetText();
    }
    else
    {
        cerr << "have not done text summary." << endl;
        return NULL;
    }
}

int XML4NLP::SetTextSummary(const char* cszTextSum)
{
    if (m_summary.nodePtr != NULL)
    {
        cerr << "has done text summary" << endl;
        return -1;
    }

    m_summary.nodePtr = new TiXmlElement(TAG_SUM);
    m_tiXmlDoc.RootElement()->LinkEndChild(m_summary.nodePtr);
    TiXmlText *textPtr = new TiXmlText(cszTextSum);
    m_summary.nodePtr->LinkEndChild(textPtr);

    return 0;
}

// ----------------------------------------------------------------for text classification
const char* XML4NLP::GetTextClass() const
{
    if (m_textclass.nodePtr != NULL)
    {
        return m_textclass.nodePtr->GetText();
    }
    else
    {
        cerr << "have not done text class." << endl;
        return NULL;
    }
}

int XML4NLP::SetTextClass(const char* cszTextClass)
{
    if (m_textclass.nodePtr != NULL)
    {
        cerr << "has done text classify" << endl;
        return -1;
    }

    m_textclass.nodePtr = new TiXmlElement(TAG_CLASS);
    m_tiXmlDoc.RootElement()->LinkEndChild(m_textclass.nodePtr);
    TiXmlText *textPtr = new TiXmlText(cszTextClass);
    m_textclass.nodePtr->LinkEndChild(textPtr);
    return 0;
}

// ----------------------------------------------------------------for SRL
int XML4NLP::CountPredArgToWord(int paragraphIdx, int sentenceIdx, int wordIdx) const
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr;
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        //cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
        //	<< " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return 0;
    }

    int counter = 0;
    do
    {
        ++counter;
        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    } while (argPtr != NULL);

    return counter;
}

int XML4NLP::CountPredArgToWord(int globalSentIdx, int wordIdx) const
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(globalSentIdx, paraIdx_sentIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words[wordIdx].wordPtr;
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        //cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
        //	<< " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return 0;
    }

    int counter = 0;
    do
    {
        ++counter;
        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    } while (argPtr != NULL);

    return counter;
}

int XML4NLP::CountPredArgToWord(int globalWordIdx) const
{
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx].sentences[sentIdx].words[wordIdx].wordPtr;
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        //cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
        //	<< " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return 0;
    }

    int counter = 0;
    do
    {
        ++counter;
        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    } while (argPtr != NULL);

    return counter;
}


int XML4NLP::GetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx,
        vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr;

    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
            << " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return -1;
    }

    if (vecType.size() != vecBegEnd.size())
    {
        cerr << "vecType's size() != vecBegEnd.size(), should resize() first." << endl;
        return -1;
    }
    if (vecType.empty())
    {
        cerr << "vecType is empty" << endl;
        return -1;
    }

    int i = 0;
    do
    {
        const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
        const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = argPtr->Attribute(TAG_END);
        vecType[i] = cszType;
        int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
        int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
        vecBegEnd[i].first = uiBeg;
        vecBegEnd[i].second = uiEnd;

        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
        ++i;
    } while (argPtr != NULL && i < vecType.size());

    if ( ! (argPtr == NULL && i == vecType.size()) )
    {
        if (argPtr == NULL)
        {
            cerr << "vecType.size() is too large" << endl;
        }
        else
        {
            cerr << "vecType.size() is too small" << endl;
        }

        return -1;
    }

    return 0;	
}

int XML4NLP::GetPredArgToWord(	int sentenceIdx, int wordIdx,
        vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words[wordIdx].wordPtr;
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
            << " of sentence " << paraIdx_sentIdx.first  << " of paragraph " << paraIdx_sentIdx.first << endl;
        return -1;
    }

    if (vecType.size() != vecBegEnd.size())
    {
        cerr << "vecType's size() != vecBegEnd.size(), should resize() first." << endl;
        return -1;
    }
    if (vecType.empty())
    {
        cerr << "vecType is empty" << endl;
        return -1;
    }

    int i = 0;
    do
    {
        const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
        const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = argPtr->Attribute(TAG_END);
        vecType[i] = cszType;
        int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
        int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
        //vecBegEnd.push_back( make_pair(uiBeg, uiEnd) );
        vecBegEnd[i].first = uiBeg;
        vecBegEnd[i].second = uiEnd;

        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
        ++i;
    } while (argPtr != NULL && i < vecType.size());

    if ( ! (argPtr == NULL && i == vecType.size()) )
    {
        if (argPtr == NULL)
        {
            cerr << "vecType.size() is too large" << endl;
        }
        else
        {
            cerr << "vecType.size() is too small" << endl;
        }

        return -1;
    }

    return 0;	
}

int XML4NLP::GetPredArgToWord(	int globalWordIdx,
        vector<const char *> &vecType, vector< pair<int, int> > &vecBegEnd) const
{
    int paraIdx, sentIdx, wordIdx;
    if (0 != MapGlobalWordIdx2paraIdx_sentIdx_wordIdx(globalWordIdx, paraIdx, sentIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx].sentences[sentIdx].words[wordIdx].wordPtr;
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
            << " of sentence " << paraIdx  << " of paragraph " << sentIdx << endl;
        return -1;
    }

    if (vecType.size() != vecBegEnd.size())
    {
        cerr << "vecType's size() != vecBegEnd.size(), should resize() first." << endl;
        return -1;
    }
    if (vecType.empty())
    {
        cerr << "vecType is empty" << endl;
        return -1;
    }

    int i = 0;
    do
    {
        const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
        const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = argPtr->Attribute(TAG_END);
        int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
        int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
        vecType[i] = cszType;
        vecBegEnd[i].first = uiBeg;
        vecBegEnd[i].second = uiEnd;

        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
        ++i;
    } while (argPtr != NULL && i < vecType.size());

    if ( ! (argPtr == NULL && i == vecType.size()) )
    {
        if (argPtr == NULL)
        {
            cerr << "vecType.size() is too large" << endl;
        }
        else
        {
            cerr << "vecType.size() is too small" << endl;
        }

        return -1;
    }

    return 0;
}

int XML4NLP::GetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx, 
        vector<string> &vecType, vector< pair<int, int> > &vecBegEnd) const
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr;

    vecType.clear();
    vecBegEnd.clear();
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
            << " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return -1;
    }

    do
    {
        const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
        const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = argPtr->Attribute(TAG_END);
        vecType.push_back(cszType != NULL ? cszType : "" );
        int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
        int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
        vecBegEnd.push_back( make_pair(uiBeg, uiEnd) );

        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    } while (argPtr != NULL);

    return 0;
}

int XML4NLP::GetPredArgToWord(	int sentenceIdx, int wordIdx, 
        vector<string> &vecType, vector< pair<int, int> > &vecBegEnd) const
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words[wordIdx].wordPtr;

    vecType.clear();
    vecBegEnd.clear();
    TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);
    if (argPtr == NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" does not exists in word " << wordIdx
            << " of sentence " << paraIdx_sentIdx.first  << " of paragraph " << paraIdx_sentIdx.first << endl;
        return -1;
    }

    do
    {
        const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
        const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = argPtr->Attribute(TAG_END);
        vecType.push_back(cszType != NULL ? cszType : "" );
        int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
        int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
        vecBegEnd.push_back( make_pair(uiBeg, uiEnd) );

        argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    } while (argPtr != NULL);
    return 0;	
}


int XML4NLP::SetPredArgToWord(	int paragraphIdx, int sentenceIdx, int wordIdx, 
        const vector<string> &vecType, const vector< pair<int, int> > &vecBegEnd)
{

    if (0 != CheckRange(paragraphIdx, sentenceIdx, wordIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words[wordIdx].wordPtr;

    if (wordPtr->FirstChildElement(TAG_SRL_ARG) != NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" already exists in word " << wordIdx
            << " of sentence " << sentenceIdx  << " of paragraph " << paragraphIdx << endl;
        return -1;
    }

    for (int i = 0; i < vecType.size(); ++i)
    {
        TiXmlElement *argPtr = new TiXmlElement(TAG_SRL_ARG);
        argPtr->SetAttribute(TAG_ID, i);
        argPtr->SetAttribute(TAG_SRL_TYPE, vecType[i].c_str());
        argPtr->SetAttribute(TAG_BEGIN, vecBegEnd[i].first);
        argPtr->SetAttribute(TAG_END, vecBegEnd[i].second);
        wordPtr->LinkEndChild(argPtr);
    }
    return 0;
}

int XML4NLP::SetPredArgToWord(	int sentenceIdx, int wordIdx, 
        const vector<string> &vecType, const vector< pair<int, int> > &vecBegEnd)
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    TiXmlElement *wordPtr = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words[wordIdx].wordPtr;

    if (wordPtr->FirstChildElement(TAG_SRL_ARG) != NULL)
    {
        cerr << "\"" << TAG_SRL_ARG << "\" already exists in word " << wordIdx
            << " of sentence " << paraIdx_sentIdx.first  << " of paragraph " << paraIdx_sentIdx.first << endl;
        return -1;
    }

    for (int i = 0; i < vecType.size(); ++i)
    {
        TiXmlElement *argPtr = new TiXmlElement(TAG_SRL_ARG);
        argPtr->SetAttribute(TAG_ID, i);
        argPtr->SetAttribute(TAG_SRL_TYPE, vecType[i].c_str());
        argPtr->SetAttribute(TAG_BEGIN, vecBegEnd[i].first);
        argPtr->SetAttribute(TAG_END, vecBegEnd[i].second);
        wordPtr->LinkEndChild(argPtr);
    }
    return 0;	
}

// ----------------------------------------------------------------for coreference resolution
int XML4NLP::GetMentionOfEntity(vector< pair<int, int> > &vecMention, int entityIdx) const
{
    if (entityIdx >= m_coref.vecEntity.size())
    {
        cerr << "entity idx is too large" << endl;
        return -1;
    }

    const vector<Mention> &vecMentionRef = m_coref.vecEntity[entityIdx].vecMention;
    if (vecMention.size() != vecMentionRef.size())
    {
        cerr << "vecMention.size() does not equal to the num of mention, should resize() first" << endl;
        return -1;
    }

    for (int i=0; i < vecMentionRef.size(); ++i)
    {
        const char *cszBeg = vecMentionRef[i].mentionPtr->Attribute(TAG_BEGIN);
        const char *cszEnd = vecMentionRef[i].mentionPtr->Attribute(TAG_END);
        if (cszBeg == NULL || cszEnd == NULL)
        {
            cerr << "mention attribute err in DOM" << endl;
            return -1;
        }
        vecMention[i].first = atoi(cszBeg);
        vecMention[i].second = atoi(cszEnd);
    }
    return 0;
}

int XML4NLP::GetCoreference(vector< vector< pair<int, int> > > &vecCoref) const
{
    if (m_coref.nodePtr == NULL)
    {
        cerr << "has not done coreference" << endl;
        return -1;
    }
    vecCoref.clear();
    TiXmlElement *crPtr = m_coref.nodePtr->FirstChildElement(TAG_COREF_CR);
    for (; crPtr != NULL; crPtr = crPtr->NextSiblingElement(TAG_COREF_CR))
    {
        vecCoref.push_back( vector< pair<int, int> >() );
        vector< pair<int, int> > &vecRef = vecCoref[vecCoref.size()-1];
        TiXmlElement *mentPtr = crPtr->FirstChildElement(TAG_COREF_MENT);
        for (; mentPtr != NULL; mentPtr = mentPtr->NextSiblingElement(TAG_COREF_MENT))
        {
            const char *cszBeg = mentPtr->Attribute(TAG_BEGIN);
            const char *cszEnd = mentPtr->Attribute(TAG_END);
            int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
            int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
            vecRef.push_back( make_pair(uiBeg, uiEnd) );
        }
    }
    return 0;
}

int XML4NLP::SetCoreference(const vector< vector< pair<int, int> > > &vecCoref)
{
    if (m_coref.nodePtr != NULL)
    {
        cerr << "has already done coreference" << endl;
        return -1;
    }

    m_coref.nodePtr = new TiXmlElement(TAG_COREF);
    for (int i = 0; i < vecCoref.size(); ++i)
    {
        TiXmlElement *crPtr = new TiXmlElement(TAG_COREF_CR);
        crPtr->SetAttribute(TAG_ID, i);

        m_coref.vecEntity.push_back( Entity() );
        Entity &entity = m_coref.vecEntity[m_coref.vecEntity.size() - 1];
        entity.entityPtr = crPtr;

        for (int j = 0; j < vecCoref[i].size(); ++j)
        {
            TiXmlElement *mentPtr = new TiXmlElement(TAG_COREF_MENT);
            mentPtr->SetAttribute(TAG_ID, j);
            mentPtr->SetAttribute(TAG_BEGIN, vecCoref[i][j].first);
            mentPtr->SetAttribute(TAG_END, vecCoref[i][j].second);
            crPtr->LinkEndChild(mentPtr);

            entity.vecMention.push_back( Mention() );			
            Mention &mention = entity.vecMention[entity.vecMention.size() - 1];
            mention.mentionPtr = mentPtr;
        }

        m_coref.nodePtr->LinkEndChild(crPtr);
    }
    m_tiXmlDoc.RootElement()->LinkEndChild(m_coref.nodePtr);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// initialize the XML DOM tree.
/// after the process LoadFile(), the DOM tree has been set up
/// but it is not fully conform to our need, 
/// for example, the member "paragraphss" of the class Document has not been initialized,
/// this function just do this.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::InitXmlStructure() {
    TiXmlElement *xml4nlp       = m_tiXmlDoc.RootElement();
    m_document.documentPtr      = xml4nlp->FirstChildElement(TAG_DOC);
    m_note.nodePtr              = xml4nlp->FirstChildElement(TAG_NOTE);
    // document summary, text classification and coreference is not
    // provided in current version (v3.0.0)
    // m_summary.nodePtr           = xml4nlp->FirstChildElement(TAG_SUM);
    // m_textclass.nodePtr         = xml4nlp->FirstChildElement(TAG_CLASS);
    // m_coref.nodePtr             = xml4nlp->FirstChildElement(TAG_COREF);

    if (m_document.documentPtr == NULL) { // consider it as wrong for now.
        cerr << "there is no \"" << TAG_DOC << "\" tag in xml file." << endl;
        return -1;
    }

    if (0 != InitXmlDocument(m_document)) {
        return -1;
    }

    if (m_coref.nodePtr != NULL) {
        if (0 != InitXmlCoref(m_coref)) {
            return -1;
        }
    }

    return 0;
}


void XML4NLP::CheckNoteForOldLtml()
{
    m_note.nodePtr = new TiXmlElement(TAG_NOTE);
    m_tiXmlDoc.RootElement()->LinkEndChild( m_note.nodePtr );
    ClearAllNote();

    //	if (m_coref.nodePtr != NULL) SetNote(NOTE_CR);
    //	if (m_summary.nodePtr != NULL) SetNote(NOTE_SUM);
    //	if (m_textclass.nodePtr != NULL) SetNote(NOTE_CLASS);

    if ( m_document.paragraphs.empty() ) return;
    if ( m_document.paragraphs[0].sentences.empty() ) return;
    SetNote(NOTE_SENT);
    if ( m_document.paragraphs[0].sentences[0].words.empty() ) return;
    SetNote(NOTE_WORD);
    TiXmlElement *wordPtr = m_document.paragraphs[0].sentences[0].words[0].wordPtr;
    if ( wordPtr->Attribute(TAG_POS) != NULL ) SetNote(NOTE_POS);
    if ( wordPtr->Attribute(TAG_NE) != NULL ) SetNote(NOTE_NE);
    if ( wordPtr->Attribute(TAG_WSD) != NULL ) SetNote(NOTE_WSD);			// consider only one attribute, excluding TAG_WSD_EXP
    if ( wordPtr->Attribute(TAG_PSR_PARENT) != NULL ) SetNote(NOTE_PARSER); // excluding TAG_PSR_RELATE
    if ( wordPtr->Attribute(TAG_SRL_ARG) != NULL ) SetNote(NOTE_SRL);		// excluding TAG_SRL_TYPE
}

int XML4NLP::InitXmlCoref(Coref &coref)
{
    TiXmlElement *entityPtr = coref.nodePtr->FirstChildElement(TAG_COREF_CR);
    if (entityPtr == NULL)
    {
        return 0;
    }

    do 
    {
        if (0 != InitXmlEntity(coref.vecEntity, entityPtr)) return -1;
        entityPtr = entityPtr->NextSiblingElement(TAG_COREF_CR);
    } while (entityPtr != NULL);
    return 0;
}

int XML4NLP::InitXmlEntity(vector<Entity> &vecEntity, TiXmlElement *entityPtr)
{
    vecEntity.push_back( Entity() );
    Entity &entity = vecEntity[vecEntity.size()-1];
    entity.entityPtr = entityPtr;

    TiXmlElement *mentionPtr = entityPtr->FirstChildElement(TAG_COREF_MENT);
    if (mentionPtr == NULL) return 0;

    do 
    {
        if (0 != InitXmlMention(entity.vecMention, mentionPtr)) return -1;
        mentionPtr = mentionPtr->NextSiblingElement(TAG_COREF_MENT);
    } while(mentionPtr != NULL);
    return 0;
}

int XML4NLP::InitXmlMention(vector<Mention> &vecMention, TiXmlElement *mentionPtr)
{
    vecMention.push_back( Mention() );
    vecMention[vecMention.size() -1].mentionPtr = mentionPtr;
    return 0;
}

int XML4NLP::InitXmlDocument(Document_t &document)
{
    TiXmlElement *paragraphPtr = document.documentPtr->FirstChildElement(TAG_PARA);
    if (paragraphPtr == NULL)	// consider it as wrong for now.
    {
        cerr << "there is no \"" << TAG_PARA << "\" tag in xml file." << endl;
        return -1;
    }

    do
    {
        if (0 != InitXmlParagraph(document.paragraphs, paragraphPtr)) return -1;
        paragraphPtr = paragraphPtr->NextSiblingElement(TAG_PARA);
    } while (paragraphPtr != NULL);
    return 0;
}

int XML4NLP::InitXmlParagraph(vector<Paragraph_t> &paragraphs, TiXmlElement *paragraphPtr)
{
    paragraphs.push_back( Paragraph_t() );
    Paragraph_t &paragraph = paragraphs[paragraphs.size()-1];
    paragraph.paragraphPtr = paragraphPtr;

    TiXmlElement *stnsPtr = paragraphPtr->FirstChildElement(TAG_SENT);
    if (stnsPtr == NULL) return 0;	// have not split sentence

    // record the sentence info
    do {
        if (0 != InitXmlSentence(paragraph.sentences, stnsPtr)) return -1;
        stnsPtr = stnsPtr->NextSiblingElement(TAG_SENT);
    } while(stnsPtr != NULL);

    return 0;
}

int XML4NLP::InitXmlSentence(vector<Sentence_t> &sentences, TiXmlElement *stnsPtr)
{
    sentences.push_back( Sentence_t() );
    Sentence_t &sentence = sentences[sentences.size()-1];
    sentence.sentencePtr = stnsPtr;

    TiXmlElement *wordPtr = stnsPtr->FirstChildElement(TAG_WORD);
    if (wordPtr == NULL) return 0;	// have not done word segment

    do
    {
        if (0 != InitXmlWord(sentence.words, wordPtr)) return -1;
        wordPtr = wordPtr->NextSiblingElement(TAG_WORD);
    } while(wordPtr != NULL);

    return 0;
}

int XML4NLP::InitXmlWord(vector<Word_t> &words, TiXmlElement *wordPtr)
{
    words.push_back( Word_t() );
    words[words.size()-1].wordPtr = wordPtr;
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// build the initial DOM tree frame.
/// it creates the XML declaration and the XSL declaration instructions and creates 
/// a root element "xml4nlp" and a child node "doc".
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::BuildDOMFrame() {
    TiXmlDeclaration * xmlDeclaration   = new TiXmlDeclaration("1.0", "utf-8", "");
    TiXmlElement * xml4nlp              = new TiXmlElement("xml4nlp");
    m_note.nodePtr                      = new TiXmlElement(TAG_NOTE);
    m_document.documentPtr            = new TiXmlElement(TAG_DOC);

    m_tiXmlDoc.LinkEndChild(xmlDeclaration);
    m_tiXmlDoc.LinkEndChild(xml4nlp);

    xml4nlp->LinkEndChild(m_note.nodePtr);
    ClearAllNote();
    xml4nlp->LinkEndChild(m_document.documentPtr);

    return 0;
}

bool XML4NLP::LTMLValidation() {
    // there should not be any attributes in `<xml4nlp>`
    // but it wont matter

    // is the attributes in `note` legal
    int state = 0;
    state |= QueryNote(NOTE_SRL);       state <<= 1;
    state |= QueryNote(NOTE_NE);        state <<= 1;
    state |= QueryNote(NOTE_PARSER);    state <<= 1; 
    state |= QueryNote(NOTE_POS);       state <<= 1;
    state |= QueryNote(NOTE_WORD);      state <<= 1;
    state |= QueryNote(NOTE_SENT);

    if (0 == state ||           //     0
            0x01 == state ||    //     1
            0x03 == state ||    //    11
            0x07 == state ||    //   111
            0x0f == state ||    //  1111
            0x17 == state ||    // 10111
            0x1f == state ||    // 11111
            0x3f == state) {
    } else {
        return false;
    }

    // if sent attribute in note is `y`, there should be an `cont`
    // attribute in para node.
    // travel through all the `para` node, query if there is a `cont`
    // attribute
    if (!(state & 0x01)) {
        for (unsigned i = 0; i < m_document.paragraphs.size(); ++ i) {
            const Paragraph_t & paragraph = m_document.paragraphs[i];
            if (!paragraph.sentences.size()) {
                if (!paragraph.paragraphPtr->GetText()) { return false; }
            } else {
                for (unsigned j = 0; j < paragraph.sentences.size(); ++ j) {
                    const Sentence_t & sentence = paragraph.sentences[j];
                    if (!sentence.sentencePtr->Attribute(TAG_CONT)) { return false; }
                }
            }
        }
    }

#define FOREACH(p, s, w) \
    for (unsigned i = 0; i < m_document.paragraphs.size(); ++ i) { \
        const Paragraph_t & p = m_document.paragraphs[i]; \
        for (unsigned j = 0; j < p.sentences.size(); ++ j) { \
            const Sentence_t & s = p.sentences[j]; \
            for (unsigned k = 0; k < s.words.size(); ++ k) { \
                const Word_t & w = s.words[k];

#define END }}}

    FOREACH(p, s, w) 
        // segment check
        if ((state & 0x02) && (!w.wordPtr->Attribute(TAG_CONT)))        { return false; }
        if ((state & 0x04) && (!w.wordPtr->Attribute(TAG_POS)))         { return false; }
        if ((state & 0x08) && (!w.wordPtr->Attribute(TAG_PSR_PARENT)))  { return false; }
        if ((state & 0x08) && (!w.wordPtr->Attribute(TAG_PSR_RELATE)))  { return false; }
        if ((state & 0x10) && (!w.wordPtr->Attribute(TAG_NE)))          { return false; }
    END

#undef END
#undef FOREACH

    return true;
}

void XML4NLP::ClearAllNote() {
    ClearNote(NOTE_SENT);
    ClearNote(NOTE_WORD);
    ClearNote(NOTE_POS);
    ClearNote(NOTE_NE);
    ClearNote(NOTE_PARSER);
    ClearNote(NOTE_WSD);
    ClearNote(NOTE_SRL);
    //	ClearNote(NOTE_CLASS);
    //	ClearNote(NOTE_SUM);
    //	ClearNote(NOTE_CR);
}

/////////////////////////////////////////////////////////////////////////////////////
/// build the paragraph structure in the DOM tree.
/// in the initial, a paragraph has only one sentence.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::BuildParagraph(string& strParagraph, int paragraphIdx) {
    if (strParagraph == "床注疑意举版低权"
            || strParagraph == "巧龙油招巴笨"
            || strParagraph == "朝注千意两版轻权") {
        strParagraph = "欢迎使用哈尔滨工业大学信息检索研究室语言技术平台！";
    } else {
    }

    TiXmlElement * documentPtr = m_document.documentPtr;
    vector<Paragraph_t> &paragraphs = m_document.paragraphs;

    paragraphs.push_back( Paragraph_t() );
    Paragraph_t &paragraph = paragraphs[paragraphs.size() - 1];

    paragraph.paragraphPtr = new TiXmlElement(TAG_PARA);
    paragraph.paragraphPtr->SetAttribute(TAG_ID, paragraphIdx);
    documentPtr->LinkEndChild(paragraph.paragraphPtr);

    TiXmlText *textPtr = new TiXmlText(strParagraph.c_str());
    paragraph.paragraphPtr->LinkEndChild( textPtr );

    return 0;
}

int XML4NLP::GetInfoFromSentence(vector<const char *> &vecInfo, 
        int paragraphIdx, 
        int sentenceIdx, 
        const char *attrName) const
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;

    const vector<Word_t> &words = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words;

    /*
       if (vecInfo.size() != words.size())
       {
    //		cerr << "vecInfo's size does not equal to word num in the sentence, should resize() first" << endl;
    return -1;
    }
    */

    if (words[0].wordPtr->Attribute(attrName) == NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" does not exists in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        vecInfo.push_back(words[i].wordPtr->Attribute(attrName));
        //vecInfo[i] = words[i].wordPtr->Attribute(attrName);
    }
    return 0;
}

int XML4NLP::GetInfoFromSentence(vector<const char *> &vecInfo, int sentenceIdx, const char *attrName) const
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    const vector<Word_t> &words = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words;

    /*
       if (vecInfo.size() != words.size())
       {
    //		cerr << "vecInfo's size does not equal to word num in the sentence, should resize() first" << endl;
    return -1;
    }
    */

    if (words[0].wordPtr->Attribute(attrName) == NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" does not exists in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        vecInfo.push_back(words[i].wordPtr->Attribute(attrName));
        //vecInfo[i] = words[i].wordPtr->Attribute(attrName);
    }
    return 0;

}


int XML4NLP::GetInfoFromSentence(vector<string> &vecInfo, int paragraphIdx, 
        int sentenceIdx, const char* attrName) const
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;

    const vector<Word_t> &words = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words;

    if (words[0].wordPtr->Attribute(attrName) == NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" does not exists in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }

    vecInfo.clear();
    for (int i = 0; i < words.size(); ++i)
    {
        const char *cszAttrValue = words[i].wordPtr->Attribute(attrName);
        vecInfo.push_back(cszAttrValue != NULL ? cszAttrValue : "");
    }
    return 0;
}

int XML4NLP::GetInfoFromSentence(vector<string> &vecInfo, int sentenceIdx, const char* attrName) const
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    const vector<Word_t> &words = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words;

    if (words[0].wordPtr->Attribute(attrName) == NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" does not exists in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }

    vecInfo.clear();
    for (int i = 0; i < words.size(); ++i)
    {
        const char *cszAttrValue = words[i].wordPtr->Attribute(attrName);
        vecInfo.push_back(cszAttrValue != NULL ? cszAttrValue : "");
    }
    return 0;
}

int XML4NLP::SetInfoToSentence(const vector<string> &vecInfo, int paragraphIdx, 
        int sentenceIdx, const char* attrName)
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;

    vector<Word_t> &words = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words;

    if (words.size() != vecInfo.size())
    {
        //		cerr << "word number does not equal to attribute \"" << attrName << "\" num in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(attrName) != NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" already exists in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        words[i].wordPtr->SetAttribute(attrName, vecInfo[i].c_str());
    }
    return 0;
}

int XML4NLP::SetInfoToSentence(const vector<string> &vecInfo, int sentenceIdx, const char* attrName)
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    vector<Word_t> &words = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words;
    if (words.size() != vecInfo.size())
    {
        //		cerr << "word number does not equal to attribute \"" << attrName << "\" num in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(attrName) != NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" already exists in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        words[i].wordPtr->SetAttribute(attrName, vecInfo[i].c_str());
    }
    return 0;
}

int XML4NLP::SetInfoToSentence(const vector<int> &vecInfo, int paragraphIdx, 
        int sentenceIdx, const char* attrName)
{
    if (0 != CheckRange(paragraphIdx, sentenceIdx)) return -1;

    vector<Word_t> &words = m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words;

    if (words.size() != vecInfo.size())
    {
        //		cerr << "word number does not equal to attribute \"" << attrName << "\" num in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(attrName) != NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" already exists in paragraph " << paragraphIdx
        //			<< " sentence " << sentenceIdx << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        words[i].wordPtr->SetAttribute(attrName, vecInfo[i]);
    }
    return 0;
}

int XML4NLP::SetInfoToSentence(const vector<int> &vecInfo, int sentenceIdx, const char* attrName)
{
    pair<int, int> paraIdx_sentIdx;
    if (0 != MapGlobalSentIdx2paraIdx_sentIdx(sentenceIdx, paraIdx_sentIdx)) return -1;

    vector<Word_t> &words = m_document.paragraphs[paraIdx_sentIdx.first].sentences[paraIdx_sentIdx.second].words;
    if (words.size() != vecInfo.size())
    {
        //		cerr << "word number does not equal to attribute \"" << attrName << "\" num in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }
    if (words[0].wordPtr->Attribute(attrName) != NULL)
    {
        //		cerr << "Attribute \"" << attrName << "\" already exists in paragraph " << paraIdx_sentIdx.first
        //			<< " sentence " << paraIdx_sentIdx.second << endl;
        return -1;
    }

    for (int i = 0; i < words.size(); ++i)
    {
        words[i].wordPtr->SetAttribute(attrName, vecInfo[i]);
    }
    return 0;
}


int XML4NLP::CheckRange(int paragraphIdx, int sentenceIdx, int wordIdx) const
{
    if (paragraphIdx >= m_document.paragraphs.size())
    {
        //		cerr << "paragraphIdx is too large: " << paragraphIdx << endl;
        return -1;
    }
    if (sentenceIdx >= m_document.paragraphs[paragraphIdx].sentences.size())
    {
        //		cerr << "sentenceIdx is too large: " << sentenceIdx << " in paragraph : " << paragraphIdx << endl;
        return -1;
    }
    if (wordIdx >= m_document.paragraphs[paragraphIdx].sentences[sentenceIdx].words.size())
    {
        //		cerr << "wordIdx is too large: " << wordIdx << " in sentence : " << sentenceIdx
        //			<< " of paragraph : " << paragraphIdx << endl;
        return -1;
    }
    return 0;
}

int XML4NLP::CheckRange(int paragraphIdx, int sentenceIdx) const
{
    if (paragraphIdx >= m_document.paragraphs.size())
    {
        //		cerr << "paragraphIdx is too large: " << paragraphIdx << endl;
        return -1;
    }
    if (sentenceIdx >= m_document.paragraphs[paragraphIdx].sentences.size())
    {
        //		cerr << "sentenceIdx is too large: " << sentenceIdx << " in paragraph : " << paragraphIdx << endl;
        return -1;
    }
    return 0;
}

int XML4NLP::CheckRange(int paragraphIdx) const {
    if (paragraphIdx >= m_document.paragraphs.size()) {
        //		cerr << "paragraphIdx is too large: " << paragraphIdx << endl;
        return -1;
    }
    return 0;
}

bool XML4NLP::QueryNote(const char *cszNoteName)  const {
    if (m_note.nodePtr == NULL) return false; // OK?

    return (strcmp(m_note.nodePtr->Attribute(cszNoteName), "y") == 0) ? true : false;
}

int XML4NLP::SetNote(const char *cszNoteName) {
    if (m_note.nodePtr == NULL) {
        m_note.nodePtr = new TiXmlElement(TAG_NOTE);
        m_tiXmlDoc.RootElement()->LinkEndChild( m_note.nodePtr );
    }
    m_note.nodePtr->SetAttribute(cszNoteName, "y");
    return 0;
}

int XML4NLP::ClearNote(const char *cszNoteName) {
    if (m_note.nodePtr == NULL) {
        m_note.nodePtr = new TiXmlElement(TAG_NOTE);
        m_tiXmlDoc.RootElement()->LinkEndChild( m_note.nodePtr );
    }

    m_note.nodePtr->SetAttribute(cszNoteName, "n");
    return 0;
}


