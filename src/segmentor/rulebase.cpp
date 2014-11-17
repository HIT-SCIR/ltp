#include "segmentor/rulebase.h"

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

  start = sent.begin();
  end = sent.end();

  while (boost::regex_search(start, end, what, uripattern, boost::match_default)) {
    int left = what[0].first - sent.begin();
    int right = what[0].second - sent.begin();

    if (flags_clear_check(flags, left, right)) {
      flags[left] = URI_BEG;
      flags_color(flags, left+1, right, URI_MID);
    }

    start = what[0].second;
  }

  start = sent.begin();
  end   = sent.end();

  while (boost::regex_search(start, end, what, engpattern, boost::match_default)) {
    int left = what[0].first - sent.begin();
    int right = what[0].second - sent.begin();

    if (flags_clear_check(flags, left, right)) {
      flags[left] = ENG_BEG;
      flags_color(flags, left+1, right, ENG_MID);
    }

    start = what[0].second;
  }

  std::string form = "";
  unsigned left  = 0;

  for (int i = 0; i < len; ) {
    int flag = 0;
    if ((flag = flags[i])) {
      form = "";

      for (; i<len && flags[i]; ++ i) {
        form += sent[i];
      }
      raw_forms.push_back(form);

      if (flag == ENG_BEG) {
        forms.push_back( __eng__ );
        if (chartypes.size() > 0) {
          chartypes.back() |= HAVE_ENG_ON_RIGHT;
        }

        chartypes.push_back(CHAR_ENG);
        chartypes.back() |= left;
        left = HAVE_ENG_ON_LEFT;
      } else if (flag == URI_BEG) {
        forms.push_back( __uri__ );
        if (chartypes.size() > 0) {
          chartypes.back() |= HAVE_URI_ON_RIGHT;
        }

        chartypes.push_back(CHAR_URI);
        chartypes.back() |= left;
        left = HAVE_URI_ON_LEFT;
      }
      ++ ret;
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
