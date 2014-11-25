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
#define MAX_WORDS_NUM  100

enum ErrorCodes {
  kEmptyStringError = 1,  /*< The input sentence is empty */
  kSplitSentenceError,    /*< Failed to perform split sentence */
  kWordsegError,          /*< Failed to perform wordseg */
  kPostagError,           /*< Failed to perform postag  */
  kParserError,           /*< Failed to perform parsing */
  kNERError,              /*< Failed to perform NER     */
  kSRLError,              /*< Failed to perform SRL     */
  kEncodingError,         /*< Sentence encoding not in UTF-8 */
  kXmlParseError,         /*< Input xml is not well formatted */
  kSentenceTooLongError,  /*< More than 300 characters or 70 words */
  kReadXmlError,          /*< Failed to read XML in internal process */
  kWriteXmlError,         /*< Failed to write XML in internal process */
};

class LTP {
public:

  /*
   * the constructor with config filepath specified to `conf/ltp.cnf`
   */
  LTP();

  /*
   * the another constructor with user specified config file
   *
   *  @param[in]  cfg_file  the path to the config file
   */
  LTP(const char * cfg_file);

  /*
   * deallocate the ltp resource
   */
  ~LTP();

  /*
   * return true on the resource successful loaded, otherwise false
   */
  bool loaded();

  // discard
  // int CreateDOMFromTxt(const char * cszTxtFileName, XML4NLP& m_xml4nlp);

  // discard
  // int CreateDOMFromXml(const char * cszXmlFileName, XML4NLP& m_xml4nlp);

  // save dom tree
  // int SaveDOM(const char *cszSaveFileName, XML4NLP& m_xml4nlp);

  /*
   * do word segmentation.
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  int wordseg(XML4NLP & xml);
  int customized_wordseg(XML4NLP & xml, const char * model_path, const char * lexicon_path);

  /*
   * do postagging
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  int postag(XML4NLP & xml);

  /*
   * do name entities recognization
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  int ner(XML4NLP & xml);

  /*
   * do dependency parsing
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  int parser(XML4NLP & xml);

  /*
   * do semantic role labeling
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  int srl(XML4NLP & xml);

  int splitSentence_dummy(XML4NLP & xml);
private:

  /*
   * split the sentence
   *
   *  @param[in/out]  xml   the xml storing ltp result
   *  @return         int   0 on success, otherwise -1
   */
  //int splitSentence_dummy(XML4NLP & xml);

  /*
   * parse the config file, and load resource according the config
   *
   *  @param[in]  confFileName  the config file
   *  @return     int           0 on success, otherwise -1
   */
  int ReadConfFile(const char *confFileName = "conf/ltp.cnf");

private:
  LTPResource m_ltpResource;    /*< the ltp resources */
  bool        m_loaded;         /*< use to sepcify if the resource is loaded */
};

#endif  //  end for __LTP_H__
