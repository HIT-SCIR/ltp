#ifndef __LTP_H__
#define __LTP_H__

#include "LTPResource.h"
#include "xml4nlp/Xml4nlp.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cassert>

#define MAX_SENTENCE_LEN 1024
#define MAX_WORDS_NUM 256

#define LTP_SERVICE_NAME_SEGMENT  "ws"
#define LTP_SERVICE_NAME_POSTAG   "pos"
#define LTP_SERVICE_NAME_NER      "ner"
#define LTP_SERVICE_NAME_DEPPARSE "dp"
#define LTP_SERVICE_NAME_SRL      "srl"

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
  static const int kActiveSegmentor = 1<<1;
  static const int kActivePostagger = 1<<2;
  static const int kActiveNER       = 1<<3;
  static const int kActiveParser    = 1<<4;
  static const int kActiveSRL       = 1<<5;

public:
  LTP(const std::string& last_stage,
      const std::string& segmentor_model_file,
      const std::string& segmentor_lexicon_file,
      const std::string& postagger_model_file,
      const std::string& postagger_lexicon_file,
      const std::string& ner_model_file,
      const std::string& parser_model_file,
      const std::string& srl_model_dir);

  ~LTP();  //! The deallocator
  bool loaded() const;  //! return true on the resource successful loaded, otherwise false

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
   * parse the config file, and load resource according the config
   *
   *  @param[in]  confFileName  the config file
   *  @return     int           0 on success, otherwise -1
   */
  bool load(const std::string& last_stage,
      const std::string& segmentor_model_file,
      const std::string& segmentor_lexicon_file,
      const std::string& postagger_model_file,
      const std::string& postagger_lexicon_file,
      const std::string& ner_model_file,
      const std::string& parser_model_file,
      const std::string& srl_model_dir);

private:
  LTPResource _resource;    /*< the ltp resources */
  bool        _loaded;         /*< use to sepcify if the resource is loaded */
};

#endif  //  end for __LTP_H__
