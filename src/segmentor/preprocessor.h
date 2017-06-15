#ifndef __LTP_SEGMENTOR_RULE_BASED_H__
#define __LTP_SEGMENTOR_RULE_BASED_H__

#include <iostream>
#include <bitset>
#include <vector>

#include "segmentor/settings.h"
#include "utils/strutils.hpp"
#include "utils/sbcdbc.hpp"
#include "utils/smartmap.hpp"
#include "utils/chartypes.hpp"

#include "boost/regex.hpp"

namespace ltp {
namespace segmentor {

class Preprocessor {
private:
  enum PreprocessFlag {
    NONE = 0,
    URI_BEG,
    URI_MID,
    URI_END,
    ENG_BEG,
    ENG_MID,
    ENG_END,
    SPECIAL_TOKEN_BEG,
    SPECIAL_TOKEN_MID,
    SPECIAL_TOKEN_END
  };

  static int HAS_SPACE_ON_LEFT;
  static int HAS_SPACE_ON_RIGHT;
  static int HAS_ENG_ON_LEFT;
  static int HAS_ENG_ON_RIGHT;
  static int HAS_URI_ON_LEFT;
  static int HAS_URI_ON_RIGHT;

  boost::regex eng_regex;
  boost::regex uri_regex;

public:
  enum PrerecognizedChartype {
    CHAR_ENG = strutils::chartypes::CHAR_PUNC+ 1,
    CHAR_URI = strutils::chartypes::CHAR_PUNC+ 2
  };

  Preprocessor();

  /**
   * preprocess the sentence
   *  @param[in]  sentence  the input sentence
   *  @param[out]  raw_forms raw characters of the input sentence
   *  @param[out]  forms  characters after preprocessing
   *  @param[out]  chartypes  character types
   */
  int preprocess(const std::string& sentence,
      std::vector<std::string>& raw_forms,
      std::vector<std::string>& forms,
      std::vector<int>& chartypes) const;

private:
  bool check_flags(const std::vector<int>& flags,
      const size_t& from,
      const size_t& to,
      const PreprocessFlag& flag) const;

  void set_flags(std::vector<int>& flags,
      const size_t& left,
      const size_t& right,
      const PreprocessFlag& flag) const;

  void special_token(const std::string& sentence,
      std::vector<int>& flags) const;

  void English(const std::string& sentence,
      std::vector<int>& flags) const;

  void URI(const std::string& sentence,
      std::vector<int>& flags) const;

  void merge(const std::string& sentence,
      const size_t& len,
      const std::vector<int>& flags,
      const PreprocessFlag& MID,
      const PreprocessFlag& END,
      const int& RIGHT_STATUS,
      const int& LEFT_STATUS,
      const std::string& ch,
      const int& chartype,
      size_t& i,
      int& left_status,
      std::vector<std::string>& raw_forms,
      std::vector<std::string>& forms,
      std::vector<int>& chartypes) const;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp 

#endif  //  end for __LTP_SEGMENTOR_RULE_BASE_H__
