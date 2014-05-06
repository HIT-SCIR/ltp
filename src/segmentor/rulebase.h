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

#if _WIN32
// disable auto-link feature in boost
#define BOOST_ALL_NO_LIB
#endif

#include "boost/regex.hpp"

namespace ltp {
namespace segmentor {
namespace rulebase {

enum { URI_BEG=1, URI_MID, URI_END, ENG_BEG, ENG_MID, ENG_END };
const int CHAR_ENG = strutils::chartypes::CHAR_PUNC+1;
const int CHAR_URI = strutils::chartypes::CHAR_PUNC+2;

const unsigned HAVE_SPACE_ON_LEFT  = (1<<3);
const unsigned HAVE_SPACE_ON_RIGHT = (1<<4);
const unsigned HAVE_ENG_ON_LEFT  = (1<<5);
const unsigned HAVE_ENG_ON_RIGHT   = (1<<6);
const unsigned HAVE_URI_ON_LEFT  = (1<<7);
const unsigned HAVE_URI_ON_RIGHT   = (1<<8);

static boost::regex engpattern("(([A-Za-z]+)([\\-'\\.][A-Za-z]+)*)");
static boost::regex uripattern("((https?|ftp|file)"
    "://[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|])");

static bool flags_clear_check(int * flags, int left, int right) {
  for (int i = left; i < right; ++ i) {
    if (flags[i]) return false;
  }
  return true;
}


static void flags_color(int * flags, int left, int right, int color) {
  for (int i = left; i < right; ++ i) {
    flags[i] = color;
  }
}

inline int preprocess(const std::string & sentence,
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

class RuleBase {
public:
  RuleBase(utility::IndexableSmartMap & labels, int style = 4) {
    // only 4 tag style is supported
    if (style == 4) {
      __trans__ = 0;
      __s_idx__ = labels.index( __s__ );
      __b_idx__ = labels.index( __b__ );
      __i_idx__ = labels.index( __i__ );
      __e_idx__ = labels.index( __e__ );

      if (__s_idx__>=0 && __b_idx__>=0 && __i_idx__>=0 && __e_idx__>=0) {
        __trans__ |= (1<<((__s_idx__<<2) + __s_idx__));
        __trans__ |= (1<<((__s_idx__<<2) + __b_idx__));
        __trans__ |= (1<<((__b_idx__<<2) + __i_idx__));
        __trans__ |= (1<<((__b_idx__<<2) + __e_idx__));
        __trans__ |= (1<<((__i_idx__<<2) + __i_idx__));
        __trans__ |= (1<<((__i_idx__<<2) + __e_idx__));
        __trans__ |= (1<<((__e_idx__<<2) + __s_idx__));
        __trans__ |= (1<<((__e_idx__<<2) + __b_idx__));
      } else {
        __trans__ = 0xffff;
      }
    }
  }

  ~RuleBase() {
  }

  inline bool legal_trans(int prev, int curr) {
    return (__trans__ & (1<<((prev<<2) + curr))) > 0;
  }

  inline bool legal_emit(int type, int curr) {
    if (((type & 0x07) == CHAR_ENG) || ((type & 0x07) == CHAR_URI)) {
      return (curr == __s_idx__);
    }

    /*if ((type & HAVE_SPACE_ON_LEFT)) {
      return (curr == __s_idx__ || curr == __b_idx__);
    }

    if ((type & HAVE_SPACE_ON_RIGHT)) {
      return (curr == __s_idx__ || curr == __e_idx__);
    }*/

    return true;
  }
private:
  unsigned __trans__;

  int __s_idx__;
  int __b_idx__;
  int __i_idx__;
  int __e_idx__;
};

}     //  end for rulebase
}     //  end for namespace segmentor
}     //  end for namespace ltp 

#endif  //  end for __LTP_SEGMENTOR_RULE_BASE_H__
