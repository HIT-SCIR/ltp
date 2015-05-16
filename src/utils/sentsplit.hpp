#ifndef __LTP_SENTENCE_SPLIT_HPP__
#define __LTP_SENTENCE_SPLIT_HPP__

#include <iostream>
#include <vector>
#include <string>

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
      sentence.append(itx->first, 1);
      char ch = text[itx->first];
      if (ch == '\r' || ch == '\n' || ch == '!' ||
          ch == '?' || ch == ';') {
        sentences.push_back(sentence);
        sentence.clear();
      }
    } else if (itx->second == itx->first + 3) {
      std::string chunk = text.substr(itx->first, itx->second- itx->first);
      bool found_periods = false;
      if (itx->second + 6 < len) {
        std::string chunk2 = text.substr(itx->first, 9);
        if (chunk2 == __three_periods_utf8_buff__[0]
          || chunk2 == __three_periods_utf8_buff__[1]
          || chunk2 == __three_periods_utf8_buff__[2]) {
          sentence.append(chunk2);
          sentences.push_back(sentence);
          sentence.clear();
          found_periods = true;
          ++itx;
        }
      }
      if (!found_periods && itx->second + 3 < len) {
        std::string chunk2= text.substr(itx->first, 6);
        if (chunk2 == __two_periods_utf8_buff__[0]
            || chunk2 == __two_periods_utf8_buff__[1]
            || chunk2 == __two_periods_utf8_buff__[2]
            || chunk2 == __two_periods_utf8_buff__[3]
            || chunk2 == __two_periods_utf8_buff__[4]
            || chunk2 == __two_periods_utf8_buff__[5]) {
          sentence.append(chunk2);
          sentences.push_back(sentence);
          sentence.clear();
          found_periods = true;
          ++ itx;
        }
      }
      if (!found_periods) {
        if (chunk == __one_periods_utf8_buff__[0]
            || chunk == __one_periods_utf8_buff__[1]
            || chunk == __one_periods_utf8_buff__[2]
            || chunk == __one_periods_utf8_buff__[3]
            || chunk == __one_periods_utf8_buff__[4]) {
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
