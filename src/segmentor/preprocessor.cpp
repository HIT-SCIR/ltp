#include "segmentor/special_tokens.h"
#include "segmentor/preprocessor.h"
#include "segmentor/settings.h"
#include "utils/strutils.hpp"

namespace ltp {
namespace segmentor {

using strutils::trim_copy;

//int Preprocessor::CHAR_ENG = strutils::chartypes::CHAR_PUNC+1;
//int Preprocessor::CHAR_URI = strutils::chartypes::CHAR_PUNC+2;

int Preprocessor::HAS_SPACE_ON_LEFT  = (1<<3);
int Preprocessor::HAS_SPACE_ON_RIGHT = (1<<4);
int Preprocessor::HAS_ENG_ON_LEFT    = (1<<5);
int Preprocessor::HAS_ENG_ON_RIGHT   = (1<<6);
int Preprocessor::HAS_URI_ON_LEFT    = (1<<7);
int Preprocessor::HAS_URI_ON_RIGHT   = (1<<8);

Preprocessor::Preprocessor():
  eng_regex("([A-Za-z0-9\\.]*[A-Za-z\\-]((—||[\\-'\\.])[A-Za-z0-9]+)*)"),
  uri_regex("((https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|])") {
}

bool Preprocessor::check_flags(const std::vector<int>& flags,
    const size_t& from, const size_t& to, const PreprocessFlag& flag) const {
  for (size_t i = from; i < to; ++ i) {
    if (flags[i] != flag) return false;
  }
  return true;
}

void Preprocessor::set_flags(std::vector<int>& flags, const size_t& from,
    const size_t& to, const PreprocessFlag& flag) const {
  for (size_t i = from; i < to; ++ i) { flags[i] = flag; }
}

void Preprocessor::special_token(const std::string& sentence,
    std::vector<int>& flags) const {

  size_t pos = 0;
  for (size_t i = 0; i < ltp::segmentor::special_tokens_size; ++i){
    const std::string& special_token = ltp::segmentor::special_tokens[i];
    while((pos = sentence.find(special_token, pos)) != std::string::npos){
      size_t pos_end = pos + special_token.length();

      if (check_flags(flags, pos, pos_end, NONE)){
        flags[pos] = SPECIAL_TOKEN_BEG;
        if(pos_end -1 > pos){
          set_flags(flags, pos+1, pos_end-1, SPECIAL_TOKEN_MID);
          flags[pos_end-1] = SPECIAL_TOKEN_END;
        }
      }
      pos = pos_end;
    }
  }
}

void Preprocessor::URI(const std::string& sentence,
    std::vector<int>& flags) const {

  std::string::const_iterator start = sentence.begin();
  std::string::const_iterator end = sentence.end();
  boost::match_results<std::string::const_iterator> what;

  // match url in the sentence
  while (boost::regex_search(start, end, what, uri_regex, boost::match_default)) {
    int left = what[0].first - sentence.begin();
    int right = what[0].second - sentence.begin();

    if (check_flags(flags, left, right, NONE)) {
      flags[left] = URI_BEG;
      if (right-1 > left) {
        set_flags(flags, left+1, right-1, URI_MID);
        flags[right-1] = URI_END;
      }
    }

    start = what[0].second;
  }
}

void Preprocessor::English(const std::string& sentence,
    std::vector<int>& flags) const {
  std::string::const_iterator start = sentence.begin();
  std::string::const_iterator end = sentence.end();
  boost::match_results<std::string::const_iterator> what;

  // match english in the sentence
  while (boost::regex_search(start, end, what, eng_regex, boost::match_default)) {
    int left = what[0].first - sentence.begin();
    int right = what[0].second - sentence.begin();
    if (check_flags(flags, left, right, NONE)) {
      flags[left] = ENG_BEG;
      if (right-1 > left) {
        set_flags(flags, left+1, right-1, ENG_MID);
        flags[right-1] = ENG_END;
      }
    }

    start = what[0].second;
  }
}

void Preprocessor::merge(const std::string& sentence,
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
    std::vector<int>& chartypes) const {
  std::string form(1, sentence[i++]);
  for (; i < len && flags[i] == MID; ++ i) {
    form += sentence[i];
  }

  if (i < len && flags[i] == END){
    form += sentence[i++];
  }

  raw_forms.push_back(form);
  forms.push_back( ch );

  if (chartypes.size() > 0 && (chartypes.back()>>3)==0) {
    chartypes.back() |= RIGHT_STATUS;
  }

  chartypes.push_back( chartype );
  chartypes.back() |= left_status;
  left_status = LEFT_STATUS;
}

int Preprocessor::preprocess(const std::string& sentence,
    std::vector<std::string>& raw_forms,
    std::vector<std::string>& forms,
    std::vector<int>& chartypes) const {
  std::string sent = trim_copy(sentence);
  // std::cerr << sent << std::endl;

  size_t len = sent.size();
  if (0 == len) {
    return 0;
  }

  size_t ret = 0;
  std::vector<int> flags(len, NONE);

  URI(sent, flags);
  special_token(sent, flags);
  English(sent, flags);

  std::string form = "";
  int left_status = 0;

  for (size_t i = 0; i < len; ) {
    int flag = 0;

    if((flag = flags[i]) == SPECIAL_TOKEN_BEG) {
      merge(sent, len, flags, SPECIAL_TOKEN_MID, SPECIAL_TOKEN_END,
          HAS_ENG_ON_RIGHT, HAS_ENG_ON_LEFT,
          __eng__, CHAR_ENG, i, left_status, raw_forms, forms, chartypes);
      ++ ret;
    } else if((flag = flags[i]) == ENG_BEG) {
      merge(sent, len, flags, ENG_MID, ENG_END, HAS_ENG_ON_RIGHT, HAS_ENG_ON_LEFT,
          __eng__, CHAR_ENG, i, left_status, raw_forms, forms, chartypes);
      ++ ret;
    } else if ((flag = flags[i]) == URI_BEG) {
      merge(sent, len, flags, URI_MID, URI_END, HAS_URI_ON_RIGHT, HAS_URI_ON_LEFT,
          __uri__, CHAR_URI, i, left_status, raw_forms, forms, chartypes);
      ++ ret;
    } else {
      bool is_space = false;
      size_t width = 0;
      if ((sent[i]&0x80)==0) {
        if ((sent[i] == ' ') || (sent[i] == '\t')) { is_space = true; }
        width = 1;
      }
      else if ((sent[i]&0xE0)==0xC0) { width = 2; }
      else if ((sent[i]&0xF0)==0xE0) { 
          width = 3; 
          if (i + 3 <= len && sent[i] == 0xffffffe3 && sent[i + 1] == 0xffffff80 && sent[i + 2] == 0xffffff80) {
              is_space = true;
          }
      }
      else if ((sent[i]&0xF8)==0xF0) { width = 4; }
      else { return -1; }


      if (is_space) {
        left_status = HAS_SPACE_ON_LEFT;
        if (chartypes.size() > 0) { chartypes.back() |= HAS_SPACE_ON_RIGHT; }
      } else {
        raw_forms.push_back(sent.substr(i, width));
        chartypes.push_back(strutils::chartypes::chartype(raw_forms.back()));
        forms.push_back("");
        strutils::chartypes::sbc2dbc(raw_forms.back(), forms.back());
        chartypes.back() |= left_status;
        left_status = 0;
        ++ ret;
      }
      i += width;
    }
  }

  return ret;
}

} //  end for namespace segmentor
} //  end for namespace ltp
