#ifndef __LTP_SENTENCE_SPLIT_HPP__
#define __LTP_SENTENCE_SPLIT_HPP__

#include <iostream>
#include <vector>
#include <string>
#include "hasher.hpp"
#include "codecs.hpp"
#include "sentsplit.tab"

namespace ltp {
namespace Chinese {

inline int split_sentence(const std::string & text,
        std::vector<std::string> & sentences,
        int encoding = strutils::codecs::UTF8) {
  sentences.clear();
  std::string sentence; sentence.reserve(512);

  int len = text.size();
  strutils::codecs::iterator itx(text, encoding);
  for (; itx.is_good() && !itx.is_end(); ++ itx) {
    if (itx->second == itx->first + 1) {
      sentence.append(text.substr(itx->first, 1));
      char ch = text[itx->first];
      if (ch == '\r' || ch == '\n' || ch == '!' ||
          ch == '?' || ch == ';') {
        sentences.push_back(sentence);
        sentence.clear();
      }
    } else if (itx->second == itx->first + 3) {
      bool found_periods = false;
      if (itx->second + 6 <= len) {
        std::string chunk = text.substr(itx->first, 9);
        size_t hashval = utility::__Default_String_HashFunction()(chunk);
        // The following black magic number is calculated by
        // hasher()(__three_periods_utf8_buff__)
        if (hashval == __three_periods_utf8_key__[0]
            || hashval == __three_periods_utf8_key__[1]
            || hashval == __three_periods_utf8_key__[2]
            || hashval == __three_periods_utf8_key__[3]) {
          sentence.append(chunk);
          sentences.push_back(sentence);
          sentence.clear();
          found_periods = true;
          ++itx;
          ++itx;
        }
      }
      if (!found_periods && itx->second + 3 <= len) {
        std::string chunk= text.substr(itx->first, 6);
        size_t hashval = utility::__Default_String_HashFunction()(chunk);
        if (hashval == __two_periods_utf8_key__[0]
            || hashval == __two_periods_utf8_key__[1]
            || hashval == __two_periods_utf8_key__[2]
            || hashval == __two_periods_utf8_key__[3]
            || hashval == __two_periods_utf8_key__[4]
            || hashval == __two_periods_utf8_key__[5]) {
          sentence.append(chunk);
          sentences.push_back(sentence);
          sentence.clear();
          found_periods = true;
          ++ itx;
        }
      }
      if (!found_periods) {
        std::string chunk = text.substr(itx->first, 3);
        size_t hashval = utility::__Default_String_HashFunction()(chunk);
        if (hashval == __one_periods_utf8_key__[0]
            || hashval == __one_periods_utf8_key__[1]
            || hashval == __one_periods_utf8_key__[2]
            || hashval == __one_periods_utf8_key__[3]) {
          sentence.append(text.substr(itx->first,3));
          sentences.push_back(sentence);
          sentence.clear();
          found_periods = true;
        }
      }
      if (!found_periods) {
        sentence.append(text.substr(itx->first,3));
      }
    } else {
      sentence.append(text.substr(itx->first,itx->second- itx->first));
    }
  }

  if (sentence.size()!=0) {
    sentences.push_back(sentence);
  }
  return sentences.size();
}

}       //  end for namespace Chinese
}       //  end for namespace ltp

#endif  //  end for __LTP_SENTENCE_SPLIT_HPP__
