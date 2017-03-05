//
// Created by Zixiang Xu on 2017/2/27.
//

#ifndef LTP_LANGUAGE_TECHNOLOGY_PLATFORM_XML4NLP_HELPER_H
#define LTP_LANGUAGE_TECHNOLOGY_PLATFORM_XML4NLP_HELPER_H

#include "xml4nlp/Xml4nlp.h"
#include "ltp/Ltp.h"
#include "json/json.h"

namespace ltp{
namespace utility{

static std::string xml2jsonstr(const XML4NLP & xml, std::string str_type) {
  Json::Value root;

  int paragraphNum = xml.CountParagraphInDocument();

  for (int pid = 0; pid < paragraphNum; ++ pid) {
    Json::Value paragraph;

    int stnsNum = xml.CountSentenceInParagraph(pid);
    for (int sid = 0; sid < stnsNum; ++sid) {
      Json::Value sentence;

      std::vector<std::string> vecWord;
      std::vector<std::string> vecPOS;
      std::vector<std::string> vecNETag;
      std::vector<std::pair<int, std::string>> vecParse;
      std::vector<std::pair<int, std::vector<std::pair<const char *, std::pair< int, int > > > > > vecSRLResult;

      // seg
      xml.GetWordsFromSentence(vecWord, pid, sid);

      // postag
      if (str_type == LTP_SERVICE_NAME_POSTAG
          || str_type == LTP_SERVICE_NAME_NER
          || str_type == LTP_SERVICE_NAME_DEPPARSE
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetPOSsFromSentence(vecPOS, pid, sid);
      }

      // ner
      if (str_type == LTP_SERVICE_NAME_NER
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetNEsFromSentence(vecNETag, pid, sid);
      }

      // dp
      if (str_type == LTP_SERVICE_NAME_DEPPARSE
          || str_type == LTP_SERVICE_NAME_SRL
          || str_type == LTP_SERVICE_NAME_ALL) {
        xml.GetParsesFromSentence(vecParse, pid, sid);
      }


      for (int wid = 0; wid < vecWord.size(); ++wid) {
        Json::Value word;
        word["id"] = wid;
        word["cont"] = vecWord[wid];

        // postag
        if (str_type == LTP_SERVICE_NAME_POSTAG
            || str_type == LTP_SERVICE_NAME_NER
            || str_type == LTP_SERVICE_NAME_DEPPARSE
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["pos"] = vecPOS[wid];

        }

        // ner
        if (str_type == LTP_SERVICE_NAME_NER
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["ne"] = vecNETag[wid];
        }

        // dp
        if (str_type == LTP_SERVICE_NAME_DEPPARSE
            || str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          word["parent"] = vecParse[wid].first;
          word["relate"] = vecParse[wid].second;
        }

        // srl
        if (str_type == LTP_SERVICE_NAME_SRL
            || str_type == LTP_SERVICE_NAME_ALL) {
          Json::Value args;
          std::vector<std::string> vecType;
          std::vector<std::pair<int, int>> vecBegEnd;
          xml.GetPredArgToWord(pid, sid, wid, vecType, vecBegEnd);
          if (vecType.size() != 0) {
            for (int arg_id = 0; arg_id < vecType.size(); ++arg_id) {
              Json::Value arg;
              arg["id"] = arg_id;
              arg["type"] = vecType[arg_id];
              arg["beg"] = vecBegEnd[arg_id].first;
              arg["end"] = vecBegEnd[arg_id].second;
              args.append(arg);
            }
          } else {
            args.resize(0);
          }
          word["arg"] = args;
        }

        sentence.append(word);
      }

      paragraph.append(sentence);
    } // sentence
    root.append(paragraph);
  } // paragraph
  return root.toStyledString();
}

}
}

#endif //LTP_LANGUAGE_TECHNOLOGY_PLATFORM_XML4NLP_HELPER_H
