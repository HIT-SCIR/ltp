#include "Ltp.h"
#include <ctime>
#include <map>
#include <string>

#include "xml4nlp/Xml4nlp.h"
#include "splitsnt/SplitSentence.h"
#include "segmentor/segment_dll.h"
#include "postagger/postag_dll.h"
#include "parser.n/parser_dll.h"
#include "ner/ner_dll.h"
#include "srl/SRL_DLL.h"
#include "utils/codecs.hpp"
#include "utils/logging.hpp"

#if _WIN32
#pragma warning(disable: 4786 4284)
#pragma comment(lib, "segmentor.lib")
#pragma comment(lib, "postagger.lib")
#pragma comment(lib, "parser.lib")
#pragma comment(lib, "ner.lib")
#pragma comment(lib, "srl.lib")
#endif

// create a platform
LTP::LTP(const std::string& last_stage,
    const std::string& segmentor_model_file,
    const std::string& segmentor_lexicon_file,
    const std::string& postagger_model_file,
    const std::string& postagger_lexicon_file,
    const std::string& ner_model_file,
    const std::string& parser_model_file,
    const std::string& srl_model_dir)
  : _resource(), _loaded(false) {
  _loaded = load(last_stage,
      segmentor_model_file, segmentor_lexicon_file,
      postagger_model_file, postagger_lexicon_file,
      ner_model_file,
      parser_model_file,
      srl_model_dir);
}

bool LTP::load(const std::string& last_stage,
    const std::string& segmentor_model_file,
    const std::string& segmentor_lexicon_file,
    const std::string& postagger_model_file,
    const std::string& postagger_lexicon_file,
    const std::string& ner_model_file,
    const std::string& parser_model_file,
    const std::string& srl_model_file) {

  size_t target_mask = 0;
  if (last_stage == LTP_SERVICE_NAME_SEGMENT) {
    target_mask = kActiveSegmentor;
  } else if (last_stage == LTP_SERVICE_NAME_POSTAG) {
    target_mask = (kActiveSegmentor|kActivePostagger);
  } else if (last_stage == LTP_SERVICE_NAME_NER) {
    target_mask = (kActiveSegmentor|kActivePostagger|kActiveNER);
  } else if (last_stage == LTP_SERVICE_NAME_DEPPARSE) {
    target_mask = (kActiveSegmentor|kActivePostagger|kActiveParser);
  } else if (last_stage == LTP_SERVICE_NAME_SRL) {
    target_mask = (kActiveSegmentor|kActivePostagger|kActiveParser|kActiveSRL);
  } else if (last_stage == "all") {
    target_mask =
      (kActiveSegmentor|kActivePostagger|kActiveNER|kActiveParser|kActiveSRL);
  }

  size_t loaded_mask = 0;

  if (target_mask & kActiveSegmentor) {
    int ret;
    if (segmentor_lexicon_file == "") {
      ret = _resource.LoadSegmentorResource(segmentor_model_file);
    } else {
      ret = _resource.LoadSegmentorResource(segmentor_model_file, segmentor_lexicon_file);
    }
    if (0 != ret) {
      ERROR_LOG("in LTP::wordseg, failed to load segmentor resource");
      return false;
    }
    loaded_mask |= kActiveSegmentor;
  }

  if (target_mask & kActivePostagger) {
    int ret;
    if (postagger_lexicon_file == "") {
      ret = _resource.LoadPostaggerResource(postagger_model_file);
    } else {
      ret = _resource.LoadPostaggerResource(postagger_model_file, postagger_lexicon_file);
    }
    if (0 != ret) {
      ERROR_LOG("in LTP::wordseg, failed to load postagger resource");
      return false;
    }
    loaded_mask |= kActivePostagger;
  }

  if (target_mask & kActiveNER) {
    if (0 != _resource.LoadNEResource(ner_model_file)) {
      ERROR_LOG("in LTP::ner, failed to load ner resource");
      return false;
    }
    loaded_mask |= kActiveNER;
  }

  if (target_mask & kActiveParser) {
    if (0 != _resource.LoadParserResource(parser_model_file)) {
      ERROR_LOG("in LTP::parser, failed to load parser resource");
      return false;
    }
    loaded_mask |= kActiveParser;
  }

  if (target_mask & kActiveSRL) {
    if ( 0 != _resource.LoadSRLResource(srl_model_file)) {
      ERROR_LOG("in LTP::srl, failed to load srl resource");
      return false;
    }
    loaded_mask |= kActiveSRL;
  }

  if ((loaded_mask & target_mask) != target_mask) {
    ERROR_LOG("target is config but resource not loaded.");
    return false;
  }

  INFO_LOG("Resources loading finished.");

  return true;
}


LTP::~LTP() {}

bool LTP::loaded() const { return _loaded; }

// If you do NOT split sentence explicitly,
// this will be called according to dependencies among modules
int LTP::splitSentence_dummy(XML4NLP & xml) {
  if ( xml.QueryNote(NOTE_SENT) ) {
    return 0;
  }

  int paraNum = xml.CountParagraphInDocument();

  if (paraNum == 0) {
    ERROR_LOG("in LTP::splitsent, There is no paragraph in doc,");
    ERROR_LOG("you may have loaded a blank file or have not loaded a file yet.");
    return kEmptyStringError;
  }

  for (int i = 0; i < paraNum; ++i) {
    vector<string> vecSentences;
    string para;
    xml.GetParagraph(i, para);

    if (0 == SplitSentence( para, vecSentences )) {
      ERROR_LOG("in LTP::splitsent, failed to split sentence");
      return kSplitSentenceError;
    }

    // dummy
    // vecSentences.push_back(para);
    if (0 != xml.SetSentencesToParagraph(vecSentences, i)) {
      ERROR_LOG("in LTP::splitsent, failed to write sentence to xml");
      return kWriteXmlError;
    }
  }

  xml.SetNote(NOTE_SENT);
  return 0;
}

// integrate word segmentor into LTP
int LTP::wordseg(XML4NLP & xml) {
  if (xml.QueryNote(NOTE_WORD)) {
    return 0;
  }

  //
  int ret = splitSentence_dummy(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::wordseg, failed to perform split sentence preprocess.");
    return ret;
  }

  // get the segmentor pointer
  void * segmentor = _resource.GetSegmentor();
  if (0 == segmentor) {
    ERROR_LOG("in LTP::wordseg, failed to init a segmentor");
    return kWordsegError;
  }

  int stnsNum = xml.CountSentenceInDocument();

  if (0 == stnsNum) {
    ERROR_LOG("in LTP::wordseg, number of sentence equals 0");
    return kEmptyStringError;
  }

  for (int i = 0; i < stnsNum; ++ i) {
    std::string strStn = xml.GetSentence(i);
    std::vector<std::string> vctWords;

    if (ltp::strutils::codecs::length(strStn) > MAX_SENTENCE_LEN) {
      ERROR_LOG("in LTP::wordseg, input sentence is too long");
      return kSentenceTooLongError;
    }

    if (0 == segmentor_segment(segmentor, strStn, vctWords)) {
      ERROR_LOG("in LTP::wordseg, failed to perform word segment on \"%s\"",
          strStn.c_str());
      return kWordsegError;
    }

    if (0 != xml.SetWordsToSentence(vctWords, i)) {
      ERROR_LOG("in LTP::wordseg, failed to write segment result to xml");
      return kWriteXmlError;
    }
  }

  xml.SetNote(NOTE_WORD);
  return 0;
}

// integrate postagger into LTP
int LTP::postag(XML4NLP & xml) {
  if ( xml.QueryNote(NOTE_POS) ) {
    return 0;
  }

  // dependency
  int ret = wordseg(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::postag, failed to perform word segment preprocess");
    return ret;
  }

  void * postagger = _resource.GetPostagger();
  if (0 == postagger) {
    ERROR_LOG("in LTP::postag, failed to init a postagger");
    return kPostagError;
  }

  int stnsNum = xml.CountSentenceInDocument();

  if (0 == stnsNum) {
    ERROR_LOG("in LTP::postag, number of sentence equals 0");
    return kEmptyStringError;
  }

  for (int i = 0; i < stnsNum; ++i) {
    vector<string> vecWord;
    vector<string> vecPOS;

    xml.GetWordsFromSentence(vecWord, i);

    if (0 == vecWord.size()) {
      ERROR_LOG("Input sentence is empty.");
      return kEmptyStringError;
    }

    if (vecWord.size() > MAX_WORDS_NUM) {
      ERROR_LOG("Input sentence is too long.");
      return kSentenceTooLongError;
    }

    if (0 == postagger_postag(postagger, vecWord, vecPOS)) {
      ERROR_LOG("in LTP::postag, failed to perform postag on sent. #%d", i+1);
      return kPostagError;
    }

    if (xml.SetPOSsToSentence(vecPOS, i) != 0) {
      ERROR_LOG("in LTP::postag, failed to write postag result to xml");
      return kWriteXmlError;
    }
  }

  xml.SetNote(NOTE_POS);

  return 0;
}

// perform ner over xml
int LTP::ner(XML4NLP & xml) {
  if ( xml.QueryNote(NOTE_NE) ) {
    return 0;
  }

  // dependency
  int ret = postag(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::ner, failed to perform postag preprocess");
    return ret;
  }

  void * ner = _resource.GetNER();

  if (NULL == ner) {
    ERROR_LOG("in LTP::ner, failed to init a ner.");
    return kNERError;
  }

  int stnsNum = xml.CountSentenceInDocument();

  if (stnsNum == 0) {
    ERROR_LOG("in LTP::ner, number of sentence equals 0");
    return kEmptyStringError;
  }

  for (int i = 0; i < stnsNum; ++ i) {
    vector<string> vecWord;
    vector<string> vecPOS;
    vector<string> vecNETag;

    if (xml.GetWordsFromSentence(vecWord, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get words from xml");
      return kReadXmlError;
    }

    if (xml.GetPOSsFromSentence(vecPOS, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get postags from xml");
      return kNERError;
    }

    if (0 == vecWord.size()) {
      ERROR_LOG("Input sentence is empty.");
      return kEmptyStringError;
    }

    if (vecWord.size() > MAX_WORDS_NUM) {
      ERROR_LOG("Input sentence is too long.");
      return kSentenceTooLongError;
    }

    if (0 == ner_recognize(ner, vecWord, vecPOS, vecNETag)) {
      ERROR_LOG("in LTP::ner, failed to perform ner on sent. #%d", i+1);
      return kNERError;
    }

    xml.SetNEsToSentence(vecNETag, i);
  }

  xml.SetNote(NOTE_NE);
  return 0;
}

int LTP::parser(XML4NLP & xml) {
  if ( xml.QueryNote(NOTE_PARSER) ) return 0;

  int ret = postag(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::parser, failed to perform postag preprocessing");
    return ret;
  }

  void * parser = _resource.GetParser();

  if (parser == NULL) {
    ERROR_LOG("in LTP::parser, failed to init a parser");
    return kParserError;
  }

  int stnsNum = xml.CountSentenceInDocument();
  if (stnsNum == 0) {
    ERROR_LOG("in LTP::parser, number of sentences equals 0");
    return kEmptyStringError;
  }

  for (int i = 0; i < stnsNum; ++i) {
    std::vector<std::string>  vecWord;
    std::vector<std::string>  vecPOS;
    std::vector<int>          vecHead;
    std::vector<std::string>  vecRel;

    if (xml.GetWordsFromSentence(vecWord, i) != 0) {
      ERROR_LOG("in LTP::parser, failed to get words from xml");
      return kReadXmlError;
    }

    if (xml.GetPOSsFromSentence(vecPOS, i) != 0) {
      ERROR_LOG("in LTP::parser, failed to get postags from xml");
      return kReadXmlError;
    }

    if (0 == vecWord.size()) {
      ERROR_LOG("Input sentence is empty.");
      return kEmptyStringError;
    }

    if (vecWord.size() > MAX_WORDS_NUM) {
      ERROR_LOG("Input sentence is too long.");
      return kSentenceTooLongError;
    }

    if (-1 == parser_parse(parser, vecWord, vecPOS, vecHead, vecRel)) {
      ERROR_LOG("in LTP::parser, failed to perform parse on sent. #%d", i+1);
      return kParserError;
    }

    if (0 != xml.SetParsesToSentence(vecHead, vecRel, i)) {
      ERROR_LOG("in LTP::parser, failed to write parse result to xml");
      return kWriteXmlError;
    }
  }

  xml.SetNote(NOTE_PARSER);

  return 0;
}

int LTP::srl(XML4NLP & xml) {
  if ( xml.QueryNote(NOTE_SRL) ) return 0;

  // dependency
  int ret = ner(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::srl, failed to perform ner preprocess");
    return ret;
  }

  ret = parser(xml);
  if (0 != ret) {
    ERROR_LOG("in LTP::srl, failed to perform parsing preprocess");
    return ret;
  }

  int stnsNum = xml.CountSentenceInDocument();
  if (stnsNum == 0) {
    ERROR_LOG("in LTP::srl, number of sentence equals 0");
    return kEmptyStringError;
  }

  for (int i = 0; i < stnsNum; ++i) {
    vector<string>              vecWord;
    vector<string>              vecPOS;
    vector<string>              vecNE;
    vector< pair<int, string> > vecParse;
    vector< pair<int, vector< pair<string, pair< int, int > > > > > vecSRLResult;

    if (xml.GetWordsFromSentence(vecWord, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get words from xml");
      return kReadXmlError;
    }

    if (xml.GetPOSsFromSentence(vecPOS, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get postags from xml");
      return kReadXmlError;
    }

    if (xml.GetNEsFromSentence(vecNE, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get ner result from xml");
      return kReadXmlError;
    }

    if (xml.GetParsesFromSentence(vecParse, i) != 0) {
      ERROR_LOG("in LTP::ner, failed to get parsing result from xml");
      return kReadXmlError;
    }

    if (0 != srl_dosrl(vecWord, vecPOS, vecParse, vecSRLResult)) {
      ERROR_LOG("in LTP::srl, failed to perform srl on sent. #%d", i+1);
      return kSRLError;
    }

    int j = 0;
    for (; j < vecSRLResult.size(); ++j) {
      vector<string>        vecType;
      vector< pair<int, int> >  vecBegEnd;
      int k = 0;

      for (; k < vecSRLResult[j].second.size(); ++k) {
        vecType.push_back(vecSRLResult[j].second[k].first);
        vecBegEnd.push_back(vecSRLResult[j].second[k].second);
      }

      if (0 != xml.SetPredArgToWord(i, vecSRLResult[j].first, vecType, vecBegEnd)) {
        return kWriteXmlError;
      }
    }
  }

  xml.SetNote(NOTE_SRL);
  return 0;
}
