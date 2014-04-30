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
#include "__util/MyLib.h"

const char * const NOTE_SENT    = "sent";
const char * const NOTE_WORD    = "word";
const char * const NOTE_POS     = "pos";
const char * const NOTE_NE      = "ne";
const char * const NOTE_PARSER  = "parser";
const char * const NOTE_WSD     = "wsd";
const char * const NOTE_SRL     = "srl";
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
  document.documentPtr = NULL;
  note.nodePtr         = NULL;
  summary.nodePtr      = NULL;
  textclass.nodePtr    = NULL;
  coref.nodePtr        = NULL;
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
  istringstream in(strTmp);  // How to use istringstream?
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
  cerr << "location :  " << endl;
  cerr << "row :     " << m_tiXmlDoc.ErrorRow() << endl;
  cerr << "col :     " << m_tiXmlDoc.ErrorCol() << endl;
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
  return LoadXMLFromString(str.c_str());
}

/////////////////////////////////////////////////////////////////////////////////////
/// clear the DOM tree, delete all nodes that allocated before.
/////////////////////////////////////////////////////////////////////////////////////
void XML4NLP::ClearDOM() {
  m_tiXmlDoc.Clear();

  document.documentPtr = NULL;
  document.paragraphs.clear();
  note.nodePtr = NULL;
  summary.nodePtr = NULL;
  textclass.nodePtr = NULL;
  coref.nodePtr = NULL;
  coref.vecEntity.clear();

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
  return document.paragraphs.size();
}

int XML4NLP::CountSentenceInParagraph(int pid) const {
  if ( 0 != CheckRange(pid) ) return 0;
  return document.paragraphs[pid].sentences.size();
}

int XML4NLP::CountSentenceInDocument() const {
  int stnsNumInDoc = 0;
  int paragraphNum = document.paragraphs.size();
  for (int i = 0; i < paragraphNum; ++i) {
    stnsNumInDoc += document.paragraphs[i].sentences.size();
  }
  return stnsNumInDoc;
}

int XML4NLP::CountWordInSentence(int pid, int sid) const {
  if ( 0 != CheckRange(pid, sid) ) return 0;
  return document.paragraphs[pid].sentences[sid].words.size();
}

int XML4NLP::CountWordInSentence(int global_sid) const {
  int pid, sid;
  if ( 0 != DecodeGlobalId(global_sid, pid, sid) ) return 0;
  return document.paragraphs[pid].sentences[sid].words.size();
}

int XML4NLP::CountWordInParagraph(int pid) const {
  if ( 0 != CheckRange(pid) ) return -1;
  int nr_words = 0;
  int nr_sents = document.paragraphs[pid].sentences.size();

  for (int i = 0; i < nr_sents; ++ i) {
    nr_words += document.paragraphs[pid].sentences[i].words.size();
  }
  return nr_words;
}

int XML4NLP::CountWordInDocument() const {
  int nr_word = 0;
  int nr_para = document.paragraphs.size();
  for (int i = 0; i < nr_para; ++ i) {
    int nr_sent = document.paragraphs[i].sentences.size();
    for (int j = 0; j < nr_sent; ++ j) {
      nr_word += document.paragraphs[i].sentences[j].words.size();
    }
  }
  return nr_word;
}

const char * XML4NLP::GetParagraph(int pid) const {
  if (0 != CheckRange(pid)) return NULL;
  if (QueryNote(NOTE_SENT)) return NULL;

  TiXmlElement *paraPtr = document.paragraphs[pid].paragraphPtr;
  return paraPtr->GetText();
}

int XML4NLP::GetParagraph(int pid, string & str) const {
  if (0 != CheckRange(pid)) { return -1; }

  const Paragraph &paragraph = document.paragraphs[pid];

  if (paragraph.sentences.empty()) {
    str = paragraph.paragraphPtr->GetText() ;
  } else {
    str = "";
    const vector<Sentence> &sentences = paragraph.sentences;
    for (int i=0; i<sentences.size(); ++i) {
      str += sentences[i].sentencePtr->Attribute(TAG_CONT);
    }
  }

  return 0;
}

#define EXTEND_FUNCTION(return_type, function_name) \
  return_type function_name (int global_sid) const { \
    int pid, sid; \
    if (0 != DecodeGlobalId(global_sid, pid, sid)) { return NULL; } \
    return (function_name)(pid, sid); \
  }

const char* XML4NLP::GetSentence(int pid, int sid) const {
  if (0 != CheckRange(pid, sid)) return NULL;
  return document.paragraphs[pid].sentences[sid].sentencePtr->Attribute(TAG_CONT);
}

EXTEND_FUNCTION(const char *, XML4NLP::GetSentence)

#define EXTEND_FUNCTION2(return_type, function_name, tag_name, failed_return) \
return_type function_name (int pid, int sid, int wid) const { \
  if (0 != CheckRange(pid, sid, wid)) { return failed_return; } \
  return document.paragraphs[pid].sentences[sid].words[wid].wordPtr->Attribute(tag_name); \
} \
\
return_type function_name (int global_sid, int wid) const { \
  int pid, sid; \
  if (0 != DecodeGlobalId(global_sid, pid, sid)) { return failed_return; } \
  return function_name (pid, sid, wid); \
} \
\
return_type function_name (int global_wid) const { \
  int pid, sid, wid; \
  if (0 != DecodeGlobalId(global_wid, pid, sid, wid)) { return failed_return; } \
  return function_name (pid, sid, wid); \
}

EXTEND_FUNCTION2 (const char *, XML4NLP::GetWord, TAG_CONT, NULL)
EXTEND_FUNCTION2 (const char *, XML4NLP::GetPOS,  TAG_POS,  NULL)
EXTEND_FUNCTION2 (const char *, XML4NLP::GetNE,   TAG_NE,   NULL)

int XML4NLP::GetWSD(WSDResult & explanation, int pid, int sid, int wid) const {
   if (0 != CheckRange(pid, sid, wid)) return -1;
   explanation.first  = document.paragraphs[pid].sentences[sid].words[wid].wordPtr->Attribute(TAG_WSD);
   explanation.second = document.paragraphs[pid].sentences[sid].words[wid].wordPtr->Attribute(TAG_WSD_EXP);
   return 0;
}

int XML4NLP::GetParse(ParseResult & relation, int pid, int sid, int wid) const {
  if (0 != CheckRange(pid, sid, wid)) return -1;
  const char * head = document.paragraphs[pid].sentences[sid].words[wid].wordPtr->Attribute(TAG_PSR_PARENT);
  relation.first  = (head == NULL ? 0 : atoi(head));
  relation.second = document.paragraphs[pid].sentences[sid].words[wid].wordPtr->Attribute(TAG_PSR_RELATE);
  return 0;
}

#define EXTEND_FUNCTION3(return_type, function_name, output_type, failed_return) \
  return_type function_name (output_type & output, int global_sid, int wid) const { \
    int pid, sid; \
    if (0 != DecodeGlobalId(global_sid, pid, sid)) { return failed_return; } \
    return function_name(output, pid, sid, wid); \
  }\
\
  return_type function_name (output_type & output, int global_wid) const { \
    int pid, wid, sid; \
    if (0 != DecodeGlobalId(global_wid, pid, sid, wid)) { return failed_return; } \
    return function_name(output, pid, sid, wid); \
  }

EXTEND_FUNCTION3 (int, XML4NLP::GetWSD,   WSDResult,    -1)
EXTEND_FUNCTION3 (int, XML4NLP::GetParse, ParseResult,  -1)

int XML4NLP::DecodeGlobalId(int global_sid, int & pid, int & sid) const {
  int startStnsIdxOfPara = 0;
  for (pid = 0; pid < document.paragraphs.size(); ++ pid) {
    int len = document.paragraphs[pid].sentences.size();
    if (startStnsIdxOfPara + len > global_sid) {
      sid = global_sid - startStnsIdxOfPara;
      return 0;
    }
    startStnsIdxOfPara += len;
  }
  return -1;
}

int XML4NLP::DecodeGlobalId(int global_wid, int & pid, int & sid, int & wid) const {
  int startWordIdxOfStns = 0;
  for (pid = 0; pid < document.paragraphs.size(); ++ pid) {
    const vector<Sentence> &sentences = document.paragraphs[pid].sentences;
    for (sid = 0; sid < sentences.size(); ++ sid) {
      if (startWordIdxOfStns + sentences[sid].words.size() > global_wid) {
        wid = global_wid - startWordIdxOfStns;
        return 0;
      }
      startWordIdxOfStns += sentences[sid].words.size();
    }
  }
  return -1;
}

int XML4NLP::GetSentencesFromParagraph(vector<const char *> &vecSentence,
    int paragraphIdx) const {
  if (0 != CheckRange(paragraphIdx)) return -1;
  if (document.paragraphs[paragraphIdx].sentences.empty()) {
    return -1;
  }

  const vector<Sentence> & sentences = document.paragraphs[paragraphIdx].sentences;
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

  if (document.paragraphs[paragraphIdx].sentences.empty()) {
    return -1;
  }

  vecSentence.clear();
  const vector<Sentence> &sentences = document.paragraphs[paragraphIdx].sentences;
  for (int i = 0; i < sentences.size(); ++ i) {
    vecSentence.push_back( sentences[i].sentencePtr->Attribute(TAG_CONT) );
  }
  return 0;
}

int XML4NLP::SetSentencesToParagraph(const vector<string> &vecSentence, int paragraphIdx) {
  if (0 != CheckRange(paragraphIdx)) {
    return -1;
  }

  if (!document.paragraphs[paragraphIdx].sentences.empty()) {
    return -1;
  }

  Paragraph & paragraph     = document.paragraphs[paragraphIdx];
  TiXmlElement * paragraphPtr   = paragraph.paragraphPtr;
  vector<Sentence> &sentences   = paragraph.sentences;

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

    sentences.push_back( Sentence() );
    sentences[sentences.size()-1].sentencePtr = sentencePtr;
  }

  return 0;
}

#define EXTEND_FUNCTION4(return_type, function_name, tag_name) \
  return_type function_name (std::vector<const char *> & output, int pid, int sid) const { \
    return GetInfoFromSentence(output, pid, sid, tag_name); \
  } \
\
  return_type function_name (std::vector<std::string> & output, int pid, int sid) const { \
    return GetInfoFromSentence(output, pid, sid, tag_name); \
  } \
\
  return_type function_name (std::vector<const char *> & output, int global_sid) const { \
    return GetInfoFromSentence(output, global_sid, tag_name); \
  } \
\
  return_type function_name (std::vector<std::string> & output, int global_sid) const { \
    return GetInfoFromSentence(output, global_sid, tag_name); \
  }

EXTEND_FUNCTION4 (int, XML4NLP::GetWordsFromSentence, TAG_CONT);
EXTEND_FUNCTION4 (int, XML4NLP::GetPOSsFromSentence,  TAG_POS);
EXTEND_FUNCTION4 (int, XML4NLP::GetNEsFromSentence,   TAG_NE);

int XML4NLP::SetWordsToSentence(const std::vector<std::string> & input,
                                int pid,
                                int sid) {
  if (0 != CheckRange(pid, sid)) return -1;

  Sentence &sentence = document.paragraphs[pid].sentences[sid];
  if (!sentence.words.empty()) {
    return -1;
  }

  for (int i = 0; i < input.size(); ++ i) {
    TiXmlElement *wordPtr = new TiXmlElement(TAG_WORD);
    wordPtr->SetAttribute(TAG_ID, i);
    wordPtr->SetAttribute(TAG_CONT, input[i].c_str());
    sentence.sentencePtr->LinkEndChild(wordPtr);

    sentence.words.push_back( Word() );
    sentence.words[sentence.words.size() - 1].wordPtr = wordPtr;
  }
  return 0;
}

int XML4NLP::SetWordsToSentence(const std::vector<std::string> & input,
                                int global_sid) {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) { return -1; }
  SetWordsToSentence(input, pid, sid);
  return 0;
}

int XML4NLP::SetPOSsToSentence(const std::vector<std::string> & input,
                               int pid, int sid) {
  return SetInfoToSentence(input, pid, sid, TAG_POS);
}

int XML4NLP::SetPOSsToSentence(const std::vector<std::string> & input,
                               int global_sid) {
  return SetInfoToSentence(input, global_sid, TAG_POS);
}

int XML4NLP::SetNEsToSentence(const std::vector<std::string> & input,
                              int pid, int sid) {
  return SetInfoToSentence(input, pid, sid, TAG_NE);
}

int XML4NLP::SetNEsToSentence(const std::vector<std::string> & input,
                              int global_sid) {
  return SetInfoToSentence(input, global_sid, TAG_NE);
}

int XML4NLP::GetParsesFromSentence(std::vector< ParseResult > &relation,
                                   int pid, int sid) const {
  std::vector<const char *> heads;
  std::vector<const char *> deprels;

  int nr_words = CountWordInSentence(pid, sid);
  relation.resize(nr_words);

  if (0 != GetInfoFromSentence(heads, pid, sid, TAG_PSR_PARENT)) {
    return -1;
  }

  if (0 != GetInfoFromSentence(deprels, pid, sid, TAG_PSR_RELATE)) {
    return -1;
  }

  for (int i = 0; i < nr_words; ++ i) {
    relation[i].first = atoi( heads[i] );
    relation[i].second = deprels[i];
  }

  return 0;
}

int XML4NLP::GetParsesFromSentence(std::vector< ParseResult > & relation,
                                   int global_sid) const {
  std::vector<const char *> heads;
  std::vector<const char *> deprels;

  int nr_words = CountWordInSentence(global_sid);
  relation.resize(nr_words);

  heads.resize(nr_words);
  deprels.resize(nr_words);

  if (0 != GetInfoFromSentence(heads, global_sid, TAG_PSR_PARENT)) {
    return -1;
  }

  if (0 != GetInfoFromSentence(deprels, global_sid, TAG_PSR_RELATE)) {
    return -1;
  }

  for (int i = 0; i < nr_words; ++ i) {
    relation[i].first = atoi( heads[i] );
    relation[i].second = deprels[i];
  }

  return 0;
}


int XML4NLP::GetParsesFromSentence(std::vector< std::pair<int, std::string > > & relation,
                                   int pid,
                                   int sid) const {
  std::vector< ParseResult > parse;
  if (0 != GetParsesFromSentence(parse, pid, sid)) {
    return -1;
  }

  relation.resize( parse.size() );
  for (int i = 0; i < parse.size(); ++ i) {
    relation[i].first = parse[i].first;
    relation[i].second = parse[i].second;
  }
  return 0;
}

int XML4NLP::GetParsesFromSentence(std::vector< std::pair<int, std::string> > & relation,
                                   int global_sid) const {
  std::vector< ParseResult > parse;
  if (0 != GetParsesFromSentence(parse, global_sid)) {
    return -1;
  }

  relation.resize( parse.size() );
  for (int i = 0; i < parse.size(); ++ i) {
    relation[i].first  = parse[i].first;
    relation[i].second = parse[i].second;
  }

  return 0;
}

int XML4NLP::SetParsesToSentence(const std::vector< std::pair<int, std::string> > & relation,
                                 int pid, int sid) {
  if (0 != CheckRange(pid, sid)) return -1;

  std::vector<Word> & words = document.paragraphs[pid].sentences[sid].words;

  if (words.size() != relation.size()) {
    std::cerr << "word number does not equal to vecInfo's size in paragraph"
              << pid
              << " sentence "
              << sid << std::endl;
    return -1;
  }

  if (words[0].wordPtr->Attribute(TAG_PSR_PARENT) != NULL) {
    std::cerr << "Attribute \""
              << TAG_PSR_PARENT
              << "\" already exists in paragraph"
              << pid
              << " sentence "
              << sid << std::endl;
    return -1;
  }

  if (words[0].wordPtr->Attribute(TAG_PSR_RELATE) != NULL) {
    std::cerr << "Attribute \""
              << TAG_PSR_RELATE
              << "\" already exists in paragraph"
              << pid
              << " sentence "
              << sid << endl;
    return -1;
  }

  for (int i = 0; i < words.size(); ++ i) {
    words[i].wordPtr->SetAttribute(TAG_PSR_PARENT, relation[i].first);
    words[i].wordPtr->SetAttribute(TAG_PSR_RELATE, relation[i].second.c_str());
  }

  return 0;
}

int XML4NLP::SetParsesToSentence(const std::vector< std::pair<int, std::string> > & relation,
                                 int global_sid) {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return SetParsesToSentence(relation, pid, sid);
}

int XML4NLP::SetParsesToSentence(const std::vector<int> & heads,
                                 const std::vector<std::string> & deprels,
                                 int pid,
                                 int sid) {
  if (0 != SetInfoToSentence(heads,   pid, sid, TAG_PSR_PARENT)) return -1;
  if (0 != SetInfoToSentence(deprels, pid, sid, TAG_PSR_RELATE)) return -1;
  return 0;
}

int XML4NLP::SetParsesToSentence(const std::vector<int> & heads,
                                 const std::vector<std::string> & deprels,
                                 int global_sid) {
  // decreasing vecHead index
  std::vector<int> d_heads;
  for (int i = 0; i < heads.size(); ++ i) {
    d_heads.push_back( heads[i] - 1 );
    // std::cout << d_heads[i] << " " << deprels[i] << std::endl;
  }

  if (0 != SetInfoToSentence(d_heads, global_sid, TAG_PSR_PARENT)) return -1;
  if (0 != SetInfoToSentence(deprels, global_sid, TAG_PSR_RELATE)) return -1;

  // std::string buffer;
  // SaveDOM(buffer);
  // std::cout << buffer << std::endl;
  return 0;
}

const char * XML4NLP::GetTextSummary() const {
  if (summary.nodePtr != NULL) {
    return summary.nodePtr->GetText();
  } else {
    std::cerr << "have not done text summary." << std::endl;
    return NULL;
  }
}

int XML4NLP::SetTextSummary(const char* cszTextSum) {
  if (summary.nodePtr != NULL) {
    std::cerr << "has done text summary" << std::endl;
    return -1;
  }

  summary.nodePtr = new TiXmlElement(TAG_SUM);
  m_tiXmlDoc.RootElement()->LinkEndChild(summary.nodePtr);
  TiXmlText * textPtr = new TiXmlText(cszTextSum);
  summary.nodePtr->LinkEndChild(textPtr);

  return 0;
}

const char * XML4NLP::GetTextClass() const {
  if (textclass.nodePtr != NULL) {
    return textclass.nodePtr->GetText();
  } else {
    cerr << "have not done text class." << endl;
    return NULL;
  }
}

int XML4NLP::SetTextClass(const char* cszTextClass) {
  if (textclass.nodePtr != NULL) {
    cerr << "has done text classify" << endl;
    return -1;
  }

  textclass.nodePtr = new TiXmlElement(TAG_CLASS);
  m_tiXmlDoc.RootElement()->LinkEndChild(textclass.nodePtr);
  TiXmlText *textPtr = new TiXmlText(cszTextClass);
  textclass.nodePtr->LinkEndChild(textPtr);
  return 0;
}

// ----------------------------------------------------------------for SRL
int XML4NLP::CountPredArgToWord(int pid, int sid, int wid) const {
  if (0 != CheckRange(pid, sid, wid)) return -1;

  TiXmlElement *wordPtr = document.paragraphs[pid].sentences[sid].words[wid].wordPtr;
  TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);

  if (argPtr == NULL) {
    return 0;
  }

  int nr_args = 0;

  do {
    ++ nr_args;
    argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
  } while (argPtr != NULL);

  return nr_args;
}

int XML4NLP::CountPredArgToWord(int global_sid, int wid) const {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return CountPredArgToWord(pid, sid, wid);
}

int XML4NLP::CountPredArgToWord(int global_wid) const {
  int pid, sid, wid;
  if (0 != DecodeGlobalId(global_wid, pid, sid, wid)) return -1;
  return CountPredArgToWord(pid, sid, wid);
}


int XML4NLP::GetPredArgToWord(int pid,
                              int sid,
                              int wid,
                              std::vector<const char *> & role,
                              std::vector< std::pair<int, int> > & range) const {
  if (0 != CheckRange(pid, sid, wid)) return -1;

  TiXmlElement *wordPtr = document.paragraphs[pid].sentences[sid].words[wid].wordPtr;
  TiXmlElement *argPtr = wordPtr->FirstChildElement(TAG_SRL_ARG);

  if (argPtr == NULL) {
    std::cerr << "\""
              << TAG_SRL_ARG
              << "\" does not exists in word "
              << wid
              << " of sentence "
              << sid
              << " of paragraph "
              << pid << std::endl;
    return -1;
  }

  if (role.size() != range.size()) {
    std::cerr << "role's size() != range.size(), should resize() first." << std::endl;
    return -1;
  }

  if (role.empty()) {
    cerr << "role is empty" << endl;
    return -1;
  }

  int i = 0;

  do {
    const char *cszType = argPtr->Attribute(TAG_SRL_TYPE);
    const char *cszBeg = argPtr->Attribute(TAG_BEGIN);
    const char *cszEnd = argPtr->Attribute(TAG_END);
    role[i] = cszType;
    int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
    int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
    range[i].first = uiBeg;
    range[i].second = uiEnd;

    argPtr = argPtr->NextSiblingElement(TAG_SRL_ARG);
    ++i;
  } while (argPtr != NULL && i < role.size());

  if ( ! (argPtr == NULL && i == role.size()) ) {
    if (argPtr == NULL) {
      cerr << "role.size() is too large" << endl;
    } else {
      cerr << "role.size() is too small" << endl;
    }

    return -1;
  }

  return 0;
}

int XML4NLP::GetPredArgToWord(int global_sid,
                              int wid,
                              std::vector<const char *> & role,
                              std::vector< std::pair<int, int> > & range) const {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return GetPredArgToWord(pid, sid, wid, role, range);
}

int XML4NLP::GetPredArgToWord(int global_wid,
                              std::vector<const char *> & role,
                              std::vector< std::pair<int, int> > & range) const {
  int pid, sid, wid;
  if (0 != DecodeGlobalId(global_wid, pid, sid, wid)) return -1;
  return GetPredArgToWord(pid, sid, wid, role, range);
}

int XML4NLP::GetPredArgToWord(int pid,
                              int sid,
                              int wid,
                              std::vector<std::string> & role,
                              std::vector< std::pair<int, int> > & range) const {
  std::vector<const char *> role2;
  int ret = GetPredArgToWord(pid, sid, wid, role2, range);
  if (0 != ret) { return ret; }

  role.resize(role2.size());
  for (int i = 0; i < role2.size(); ++ i) { role[i] = role2[i]; }
  return 0;
}

int XML4NLP::GetPredArgToWord(int global_sid,
                              int wid,
                              std::vector<std::string> & role,
                              std::vector< std::pair<int, int> > & range) const {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return GetPredArgToWord(pid, sid, wid, role, range);
}


int XML4NLP::SetPredArgToWord(int pid,
                              int sid,
                              int wid,
                              const std::vector<std::string> & role,
                              const std::vector< std::pair<int, int> > & range) {
  if (0 != CheckRange(pid, sid, wid)) return -1;

  TiXmlElement *wordPtr = document.paragraphs[pid].sentences[sid].words[wid].wordPtr;

  if (wordPtr->FirstChildElement(TAG_SRL_ARG) != NULL) {
    std::cerr << "\""
              << TAG_SRL_ARG
              << "\" already exists in word "
              << wid
              << " of sentence "
              << sid
              << " of paragraph "
              << pid << std::endl;
    return -1;
  }

  for (int i = 0; i < role.size(); ++ i) {
    TiXmlElement *argPtr = new TiXmlElement(TAG_SRL_ARG);
    argPtr->SetAttribute(TAG_ID, i);
    argPtr->SetAttribute(TAG_SRL_TYPE, role[i].c_str());
    argPtr->SetAttribute(TAG_BEGIN, range[i].first);
    argPtr->SetAttribute(TAG_END, range[i].second);
    wordPtr->LinkEndChild(argPtr);
  }

  return 0;
}

int XML4NLP::SetPredArgToWord(int global_sid,
                              int wid,
                              const std::vector<std::string> & role,
                              const std::vector< std::pair<int, int> > & range) {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return SetPredArgToWord(pid, sid, wid, role, range);
}

int XML4NLP::GetMentionOfEntity(std::vector< std::pair<int, int> > &mention,
                                int entityIdx) const {
  if (entityIdx >= coref.vecEntity.size()) {
    cerr << "entity idx is too large" << endl;
    return -1;
  }

  const vector<Mention> &mentionRef = coref.vecEntity[entityIdx].vecMention;
  if (mention.size() != mentionRef.size()) {
    std::cerr << "mention.size() does not equal to the num of mention,"
              << " should resize() first"
              << std::endl;
    return -1;
  }

  for (int i=0; i < mentionRef.size(); ++i) {
    const char *cszBeg = mentionRef[i].mentionPtr->Attribute(TAG_BEGIN);
    const char *cszEnd = mentionRef[i].mentionPtr->Attribute(TAG_END);
    if (cszBeg == NULL || cszEnd == NULL) {
      std::cerr << "mention attribute err in DOM" << std::endl;
      return -1;
    }
    mention[i].first = atoi(cszBeg);
    mention[i].second = atoi(cszEnd);
  }
  return 0;
}

int XML4NLP::GetCoreference(vector< vector< pair<int, int> > > &vecCoref) const {
  if (coref.nodePtr == NULL) {
    cerr << "has not done coreference" << endl;
    return -1;
  }
  vecCoref.clear();
  TiXmlElement *crPtr = coref.nodePtr->FirstChildElement(TAG_COREF_CR);

  for (; crPtr != NULL; crPtr = crPtr->NextSiblingElement(TAG_COREF_CR)) {
    vecCoref.push_back( vector< pair<int, int> >() );
    vector< pair<int, int> > &vecRef = vecCoref[vecCoref.size()-1];
    TiXmlElement *mentPtr = crPtr->FirstChildElement(TAG_COREF_MENT);

    for (; mentPtr != NULL; mentPtr = mentPtr->NextSiblingElement(TAG_COREF_MENT)) {
      const char *cszBeg = mentPtr->Attribute(TAG_BEGIN);
      const char *cszEnd = mentPtr->Attribute(TAG_END);
      int uiBeg = static_cast<int>(cszBeg != NULL ? atoi(cszBeg) : 0);
      int uiEnd = static_cast<int>(cszEnd != NULL ? atoi(cszEnd) : 0);
      vecRef.push_back( make_pair(uiBeg, uiEnd) );
    }
  }
  return 0;
}

int XML4NLP::SetCoreference(const vector< vector< pair<int, int> > > &vecCoref) {
  if (coref.nodePtr != NULL) {
    cerr << "has already done coreference" << endl;
    return -1;
  }

  coref.nodePtr = new TiXmlElement(TAG_COREF);
  for (int i = 0; i < vecCoref.size(); ++i) {
    TiXmlElement *crPtr = new TiXmlElement(TAG_COREF_CR);
    crPtr->SetAttribute(TAG_ID, i);

    coref.vecEntity.push_back( Entity() );
    Entity &entity = coref.vecEntity[coref.vecEntity.size() - 1];
    entity.entityPtr = crPtr;

    for (int j = 0; j < vecCoref[i].size(); ++j) {
      TiXmlElement *mentPtr = new TiXmlElement(TAG_COREF_MENT);
      mentPtr->SetAttribute(TAG_ID, j);
      mentPtr->SetAttribute(TAG_BEGIN, vecCoref[i][j].first);
      mentPtr->SetAttribute(TAG_END, vecCoref[i][j].second);
      crPtr->LinkEndChild(mentPtr);

      entity.vecMention.push_back( Mention() );
      Mention &mention = entity.vecMention[entity.vecMention.size() - 1];
      mention.mentionPtr = mentPtr;
    }

    coref.nodePtr->LinkEndChild(crPtr);
  }
  m_tiXmlDoc.RootElement()->LinkEndChild(coref.nodePtr);

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
  TiXmlElement *xml4nlp     = m_tiXmlDoc.RootElement();
  document.documentPtr    = xml4nlp->FirstChildElement(TAG_DOC);
  note.nodePtr        = xml4nlp->FirstChildElement(TAG_NOTE);
  // document summary, text classification and coreference is not
  // provided in current version (v3.0.0)
  // summary.nodePtr       = xml4nlp->FirstChildElement(TAG_SUM);
  // textclass.nodePtr     = xml4nlp->FirstChildElement(TAG_CLASS);
  // coref.nodePtr       = xml4nlp->FirstChildElement(TAG_COREF);

  if (document.documentPtr == NULL) { // consider it as wrong for now.
    cerr << "there is no \"" << TAG_DOC << "\" tag in xml file." << endl;
    return -1;
  }

  if (0 != InitXmlDocument(document)) {
    return -1;
  }

  if (coref.nodePtr != NULL) {
    if (0 != InitXmlCoref(coref)) {
      return -1;
    }
  }

  return 0;
}

int XML4NLP::InitXmlCoref(Coref &coref) {
  TiXmlElement *entityPtr = coref.nodePtr->FirstChildElement(TAG_COREF_CR);

  if (entityPtr == NULL) {
    return 0;
  }

  do {
    if (0 != InitXmlEntity(coref.vecEntity, entityPtr)) return -1;
    entityPtr = entityPtr->NextSiblingElement(TAG_COREF_CR);
  } while (entityPtr != NULL);
  return 0;
}

int XML4NLP::InitXmlEntity(vector<Entity> &vecEntity, TiXmlElement *entityPtr) {
  vecEntity.push_back( Entity() );
  Entity &entity = vecEntity[vecEntity.size()-1];
  entity.entityPtr = entityPtr;

  TiXmlElement *mentionPtr = entityPtr->FirstChildElement(TAG_COREF_MENT);
  if (mentionPtr == NULL) return 0;

  do {
    if (0 != InitXmlMention(entity.vecMention, mentionPtr)) return -1;
    mentionPtr = mentionPtr->NextSiblingElement(TAG_COREF_MENT);
  } while(mentionPtr != NULL);
  return 0;
}

int XML4NLP::InitXmlMention(vector<Mention> &vecMention, TiXmlElement *mentionPtr) {
  vecMention.push_back( Mention() );
  vecMention[vecMention.size() -1].mentionPtr = mentionPtr;
  return 0;
}

int XML4NLP::InitXmlDocument(Document &document) {
  TiXmlElement *paragraphPtr = document.documentPtr->FirstChildElement(TAG_PARA);
  if (paragraphPtr == NULL)   {
    // consider it as wrong for now.
    cerr << "there is no \"" << TAG_PARA << "\" tag in xml file." << endl;
    return -1;
  }

  do {
    if (0 != InitXmlParagraph(document.paragraphs, paragraphPtr)) return -1;
    paragraphPtr = paragraphPtr->NextSiblingElement(TAG_PARA);
  } while (paragraphPtr != NULL);
  return 0;
}

int XML4NLP::InitXmlParagraph(vector<Paragraph> &paragraphs, TiXmlElement *paragraphPtr)
{
  paragraphs.push_back( Paragraph() );
  Paragraph &paragraph = paragraphs[paragraphs.size()-1];
  paragraph.paragraphPtr = paragraphPtr;

  TiXmlElement *stnsPtr = paragraphPtr->FirstChildElement(TAG_SENT);
  if (stnsPtr == NULL) return 0;  // have not split sentence

  // record the sentence info
  do {
    if (0 != InitXmlSentence(paragraph.sentences, stnsPtr)) return -1;
    stnsPtr = stnsPtr->NextSiblingElement(TAG_SENT);
  } while(stnsPtr != NULL);

  return 0;
}

int XML4NLP::InitXmlSentence(vector<Sentence> &sentences, TiXmlElement *stnsPtr)
{
  sentences.push_back( Sentence() );
  Sentence &sentence = sentences[sentences.size()-1];
  sentence.sentencePtr = stnsPtr;

  TiXmlElement *wordPtr = stnsPtr->FirstChildElement(TAG_WORD);
  if (wordPtr == NULL) return 0;  // have not done word segment

  do
  {
    if (0 != InitXmlWord(sentence.words, wordPtr)) return -1;
    wordPtr = wordPtr->NextSiblingElement(TAG_WORD);
  } while(wordPtr != NULL);

  return 0;
}

int XML4NLP::InitXmlWord(vector<Word> &words, TiXmlElement *wordPtr) {
  words.push_back( Word() );
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
  TiXmlElement * xml4nlp        = new TiXmlElement("xml4nlp");
  note.nodePtr            = new TiXmlElement(TAG_NOTE);
  document.documentPtr      = new TiXmlElement(TAG_DOC);

  m_tiXmlDoc.LinkEndChild(xmlDeclaration);
  m_tiXmlDoc.LinkEndChild(xml4nlp);

  xml4nlp->LinkEndChild(note.nodePtr);
  ClearAllNote();
  xml4nlp->LinkEndChild(document.documentPtr);

  return 0;
}

bool XML4NLP::LTMLValidation() {
  // there should not be any attributes in `<xml4nlp>`
  // but it wont matter
  if (!note.nodePtr->Attribute(NOTE_SENT)
      || !note.nodePtr->Attribute(NOTE_WORD)
      || !note.nodePtr->Attribute(NOTE_POS)
      || !note.nodePtr->Attribute(NOTE_PARSER)
      || !note.nodePtr->Attribute(NOTE_NE)
      || !note.nodePtr->Attribute(NOTE_SRL)) {
    return false;
  }

  // is the attributes in `note` legal
  int state = 0;
  state |= QueryNote(NOTE_SRL);     state <<= 1;
  state |= QueryNote(NOTE_NE);      state <<= 1;
  state |= QueryNote(NOTE_PARSER);  state <<= 1;
  state |= QueryNote(NOTE_POS);     state <<= 1;
  state |= QueryNote(NOTE_WORD);    state <<= 1;
  state |= QueryNote(NOTE_SENT);

  if (0 == state ||     //     0
      0x01 == state ||  //     1
      0x03 == state ||  //    11
      0x07 == state ||  //   111
      0x0f == state ||  //  1111
      0x17 == state ||  // 10111
      0x1f == state ||  // 11111
      0x3f == state) {
  } else {
    return false;
  }

  // if sent attribute in note is `y`, there should be an `cont`
  // attribute in para node.
  // travel through all the `para` node, query if there is a `cont`
  // attribute
  if (!(state & 0x01)) {
    for (unsigned i = 0; i < document.paragraphs.size(); ++ i) {
      const Paragraph & paragraph = document.paragraphs[i];
      if (!paragraph.sentences.size()) {
        if (!paragraph.paragraphPtr->GetText()) { return false; }
      } else {
        for (unsigned j = 0; j < paragraph.sentences.size(); ++ j) {
          const Sentence & sentence = paragraph.sentences[j];
          if (!sentence.sentencePtr->Attribute(TAG_CONT)) { return false; }
        }
      }
    }
  }

#define FOREACH(p, s, w) \
  for (unsigned i = 0; i < document.paragraphs.size(); ++ i) { \
    const Paragraph & p = document.paragraphs[i]; \
    for (unsigned j = 0; j < p.sentences.size(); ++ j) { \
      const Sentence & s = p.sentences[j]; \
      for (unsigned k = 0; k < s.words.size(); ++ k) { \
        const Word & w = s.words[k];

#define END }}}

  FOREACH(p, s, w) 
    // segment check
    const char * buffer = NULL;
    buffer = w.wordPtr->Attribute(TAG_CONT);
    if ((state & 0x02) 
        && (!buffer || !strnlen(buffer, 1024)))  { return false; }

    buffer = w.wordPtr->Attribute(TAG_POS);
    if ((state & 0x04)
        && (!buffer || !strnlen(buffer, 1024)))  { return false; }

    buffer = w.wordPtr->Attribute(TAG_PSR_PARENT);
    if ((state & 0x08) 
        && (!buffer || !strnlen(buffer, 1024)))  { return false; }

    buffer = w.wordPtr->Attribute(TAG_PSR_RELATE);
    if ((state & 0x08)
        && (!buffer || !strnlen(buffer, 1024)))  { return false; }

    buffer = w.wordPtr->Attribute(TAG_NE);
    if ((state & 0x10)
        && (!buffer || !strnlen(buffer, 1024)))  { return false; }
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
  //  ClearNote(NOTE_CLASS);
  //  ClearNote(NOTE_SUM);
  //  ClearNote(NOTE_CR);
}

/////////////////////////////////////////////////////////////////////////////////////
/// build the paragraph structure in the DOM tree.
/// in the initial, a paragraph has only one sentence.
/////////////////////////////////////////////////////////////////////////////////////
int XML4NLP::BuildParagraph(string& strParagraph, int paragraphIdx) {

  TiXmlElement * documentPtr = document.documentPtr;
  vector<Paragraph> &paragraphs = document.paragraphs;

  paragraphs.push_back( Paragraph() );
  Paragraph &paragraph = paragraphs[paragraphs.size() - 1];

  paragraph.paragraphPtr = new TiXmlElement(TAG_PARA);
  paragraph.paragraphPtr->SetAttribute(TAG_ID, paragraphIdx);
  documentPtr->LinkEndChild(paragraph.paragraphPtr);

  TiXmlText *textPtr = new TiXmlText(strParagraph.c_str());
  paragraph.paragraphPtr->LinkEndChild( textPtr );

  return 0;
}

int XML4NLP::GetInfoFromSentence(std::vector<const char *> & info,
                                 int pid,
                                 int sid,
                                 const char *attribute_name) const {
  if (0 != CheckRange(pid, sid)) return -1;

  const vector<Word> & words = document.paragraphs[pid].sentences[sid].words;
  if (words[0].wordPtr->Attribute(attribute_name) == NULL) {
    return -1;
  }

  info.resize(words.size());
  for (int i = 0; i < words.size(); ++ i) {
    info[i] = words[i].wordPtr->Attribute(attribute_name);
  }
  return 0;
}

int XML4NLP::GetInfoFromSentence(std::vector<const char *> & info,
                                 int global_sid,
                                 const char *attribute_name) const {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return GetInfoFromSentence(info, pid, sid, attribute_name);
}


int XML4NLP::GetInfoFromSentence(std::vector<std::string> &info,
                                 int pid,
                                 int sid,
                                 const char* attribute_name) const {
  if (0 != CheckRange(pid, sid)) return -1;

  const vector<Word> & words = document.paragraphs[pid].sentences[sid].words;

  if (words[0].wordPtr->Attribute(attribute_name) == NULL) {
    return -1;
  }

  info.clear();
  for (int i = 0; i < words.size(); ++ i) {
    const char * cszAttrValue = words[i].wordPtr->Attribute(attribute_name);
    info.push_back(cszAttrValue != NULL ? cszAttrValue : "");
  }
  return 0;
}

int XML4NLP::GetInfoFromSentence(std::vector<std::string> & info,
                                 int global_sid,
                                 const char* attribute_name) const {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;

  return GetInfoFromSentence(info, pid, sid, attribute_name);
}

int XML4NLP::SetInfoToSentence(const std::vector<std::string> & info,
                               int pid,
                               int sid,
                               const char* attribute_name) {
  if (0 != CheckRange(pid, sid)) return -1;

  std::vector<Word> & words = document.paragraphs[pid].sentences[sid].words;

  if (words.size() != info.size()) {
    return -1;
  }

  if (words[0].wordPtr->Attribute(attribute_name) != NULL) {
    return -1;
  }

  for (int i = 0; i < words.size(); ++ i) {
    // std::cout << attribute_name << " " << info[i] << std::endl;
    words[i].wordPtr->SetAttribute(attribute_name, info[i].c_str());
  }
  return 0;
}

int XML4NLP::SetInfoToSentence(const std::vector<std::string> & info,
                               int global_sid,
                               const char * attribute_name) {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;

  return SetInfoToSentence(info, pid, sid, attribute_name);
}

int XML4NLP::SetInfoToSentence(const std::vector<int> & info,
                               int pid,
                               int sid,
                               const char * attribute_name) {
  if (0 != CheckRange(pid, sid)) return -1;

  std::vector<Word> & words = document.paragraphs[pid].sentences[sid].words;

  if (words.size() != info.size()) {
    return -1;
  }

  if (words[0].wordPtr->Attribute(attribute_name) != NULL) {
    return -1;
  }

  for (int i = 0; i < words.size(); ++ i) {
    // std::cout << attribute_name << " " << info[i] << std::endl;
    words[i].wordPtr->SetAttribute(attribute_name, info[i]);
  }
  return 0;
}

int XML4NLP::SetInfoToSentence(const std::vector<int> & info,
                               int global_sid,
                               const char * attribute_name) {
  int pid, sid;
  if (0 != DecodeGlobalId(global_sid, pid, sid)) return -1;
  return SetInfoToSentence(info, pid, sid, attribute_name);
}


int XML4NLP::CheckRange(int pid, int sid, int wid) const {
  if (pid >= document.paragraphs.size()) {
    return -1;
  }

  if (sid >= document.paragraphs[pid].sentences.size()) {
    return -1;
  }

  if (wid >= document.paragraphs[pid].sentences[sid].words.size()) {
    return -1;
  }
  return 0;
}

int XML4NLP::CheckRange(int pid, int sid) const {
  if (pid >= document.paragraphs.size()) {
    return -1;
  }

  if (sid >= document.paragraphs[pid].sentences.size()) {
    return -1;
  }

  return 0;
}

int XML4NLP::CheckRange(int paragraphIdx) const {
  if (paragraphIdx >= document.paragraphs.size()) {
    return -1;
  }
  return 0;
}

bool XML4NLP::QueryNote(const char *cszNoteName)  const {
  if (note.nodePtr == NULL) return false; // OK?

  return (strcmp(note.nodePtr->Attribute(cszNoteName), "y") == 0) ? true : false;
}

int XML4NLP::SetNote(const char *cszNoteName) {
  if (note.nodePtr == NULL) {
    note.nodePtr = new TiXmlElement(TAG_NOTE);
    m_tiXmlDoc.RootElement()->LinkEndChild( note.nodePtr );
  }
  note.nodePtr->SetAttribute(cszNoteName, "y");
  return 0;
}

int XML4NLP::ClearNote(const char *cszNoteName) {
  if (note.nodePtr == NULL) {
    note.nodePtr = new TiXmlElement(TAG_NOTE);
    m_tiXmlDoc.RootElement()->LinkEndChild( note.nodePtr );
  }

  note.nodePtr->SetAttribute(cszNoteName, "n");
  return 0;
}
