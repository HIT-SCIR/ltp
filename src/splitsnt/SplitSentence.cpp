#include "splitsnt/SplitSentence.h"
#include "utils/strutils.hpp"
#include "utils/sentsplit.hpp"

using ltp::strutils::trim;

SPLIT_SENTENCE_DLL_API int SplitSentence(const std::string& text,
    std::vector<std::string>& sentences) {

  ltp::Chinese::split_sentence(text, sentences);
  for (size_t i = 0; i < sentences.size(); ++ i) {
    trim(sentences[i]);
  }
  return 1;
}
