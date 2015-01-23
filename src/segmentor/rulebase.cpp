#include "segmentor/rulebase.h"
#include "segmentor/special_tokens.h"

namespace ltp {
namespace segmentor {
namespace rulebase {

int preprocess(const std::string & sentence,
    std::vector<std::string> & raw_forms,
    std::vector<std::string> & forms,
    std::vector<int> & chartypes) {

  std::string sent = ltp::strutils::chomp(sentence);
  // std::cerr << sent << std::endl;

  int len = sent.size();
  if (0 == len) {
    return 0;
  }

  std::string::const_iterator start, end;
  boost::match_results<std::string::const_iterator> what;

  int ret = 0;
  int * flags = new int[len];

  for (int i = 0; i < len; ++ i) {
    flags[i] = 0;
  }

  int pos = 0;
  for (int i = 0; i < ltp::segmentor::special_tokens_size; ++i){
      pos = 0;
      const std::string& special_token = ltp::segmentor::special_tokens[i];
      while((pos = sent.find(special_token, pos)) != std::string::npos){
          int pos_end = pos + special_token.length();
              if (flags_clear_check(flags, pos, pos_end)){
                  flags[pos] = SPECIAL_TOKEN_BEG;
                  if(pos_end -1 > pos){
                    flags_color(flags, pos+1, pos_end-1, SPECIAL_TOKEN_MID);
                    flags[pos_end-1] = SPECIAL_TOKEN_END;
                  }
              }
          pos = pos_end;
      }
  }

  start = sent.begin();
  end = sent.end();
  
  // match url in the sentence
  while (boost::regex_search(start, end, what, uripattern, boost::match_default)) {
    int left = what[0].first - sent.begin();
    int right = what[0].second - sent.begin();

    if (flags_clear_check(flags, left, right)) {
      flags[left] = URI_BEG;
      if(right-1 > left){
        flags_color(flags, left+1, right-1, URI_MID);
        flags[right-1] = URI_END;
      }
    }

    start = what[0].second;
  }

  start = sent.begin();
  end   = sent.end();

  // match english in the sentence
  while (boost::regex_search(start, end, what, engpattern, boost::match_default)) {
    int left = what[0].first - sent.begin();
    int right = what[0].second - sent.begin();
    if (flags_clear_check(flags, left, right)) {
      flags[left] = ENG_BEG;
      if(right-1 > left){
        flags_color(flags, left+1, right-1, ENG_MID);
        flags[right-1]=ENG_END;
      }
    }

    start = what[0].second;
  }

  std::string form = "";
  unsigned left  = 0;

  for (int i = 0; i < len; ) {
    int flag = 0;

    if((flag = flags[i]) == SPECIAL_TOKEN_BEG){
      form = "";
      form += sent[i++];
      for (; i<len && flags[i]==SPECIAL_TOKEN_MID; ++ i) {
        form += sent[i];
      }
      if(i < len && flags[i]==SPECIAL_TOKEN_END){
        form += sent[i++];
      }
      raw_forms.push_back(form);
      forms.push_back( __eng__ );
        if (chartypes.size() > 0) {
          chartypes.back() |= HAVE_ENG_ON_RIGHT;
        }

        chartypes.push_back(CHAR_ENG);
        chartypes.back() |= left;
        left = HAVE_ENG_ON_LEFT;
      ++ret;
    } else if((flag = flags[i]) == ENG_BEG){
      form = "";
      form += sent[i++];
      for (; i<len && flags[i]==ENG_MID; ++ i) {
        form += sent[i];
      }
      if(i < len && flags[i]==ENG_END){
        form += sent[i++];
      }
      raw_forms.push_back(form);
        forms.push_back( __eng__ );
        if (chartypes.size() > 0) {
          chartypes.back() |= HAVE_ENG_ON_RIGHT;
        }

        chartypes.push_back(CHAR_ENG);
        chartypes.back() |= left;
        left = HAVE_ENG_ON_LEFT;
        ++ret;
    } else if ((flag = flags[i]) == URI_BEG){
      form = "";
      form += sent[i++];
      for (; i<len && flags[i]==URI_MID; ++ i) {
        form += sent[i];
      }
      if(i < len && flags[i]==URI_END){
        form += sent[i++];
      }
      raw_forms.push_back(form);
        forms.push_back( __uri__ );
        if (chartypes.size() > 0) {
          chartypes.back() |= HAVE_URI_ON_RIGHT;
        }

        chartypes.push_back(CHAR_URI);
        chartypes.back() |= left;
        left = HAVE_URI_ON_LEFT;
        ++ret;
    } else {
      if ((sent[i]&0x80)==0) {
        if ((sent[i] != ' ') && (sent[i] != '\t')) {
          raw_forms.push_back(sent.substr(i, 1));
          chartypes.push_back(strutils::chartypes::chartype(raw_forms.back()));
          forms.push_back("");
          strutils::chartypes::sbc2dbc(raw_forms.back(), forms.back());
          chartypes.back() |= left;
          left = 0;
        } else {
          left = HAVE_SPACE_ON_LEFT;
          if (chartypes.size()>0) {
            chartypes.back() |= HAVE_SPACE_ON_RIGHT;
          }
        }
        ++ i;
      } else if ((sent[i]&0xE0)==0xC0) {
        raw_forms.push_back(sent.substr(i, 2));
        chartypes.push_back(strutils::chartypes::chartype(raw_forms.back()));
        forms.push_back("");
        strutils::chartypes::sbc2dbc(raw_forms.back(), forms.back());
        chartypes.back() |= left;
        left = 0;
        i += 2;
      } else if ((sent[i]&0xF0)==0xE0) {
        raw_forms.push_back(sent.substr(i, 3));
        chartypes.push_back(strutils::chartypes::chartype(raw_forms.back()));
        forms.push_back("");
        strutils::chartypes::sbc2dbc(raw_forms.back(), forms.back());
        chartypes.back() |= left;
        i += 3;
      } else if ((sent[i]&0xF8)==0xF0) {
        raw_forms.push_back(sent.substr(i, 4));
        chartypes.push_back(strutils::chartypes::chartype(raw_forms.back()));
        forms.push_back("");
        strutils::chartypes::sbc2dbc(raw_forms.back(), forms.back());
        chartypes.back() |= left;
        i += 4;
      } else {
        delete [](flags);
        return -1;
      }

      ++ ret;
    }
  }

  delete [](flags);
  return ret;
}


} //  end for namespace rulebase
} //  end for namespace segmentor
} //  end for namespace ltp
