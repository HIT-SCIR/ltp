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

static boost::regex engpattern("([A-Za-z0-9\\.]*[A-Za-z\\-]((—||[\\-'\\.])[A-Za-z0-9]+)*)");
//static boost::regex engpattern("(([A-Za-z]+)([\\-'\\.][A-Za-z]+)*)");
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

/**
 * preprocess the sentence
 *  @param[in]  sentence  the input sentence
 *  @param[out]  raw_forms raw characters of the input sentence
 *  @param[out]  forms  characters after preprocessing
 *  @param[out]  chartypes  character types
 */
int preprocess(const std::string & sentence,
    std::vector<std::string> & raw_forms,
    std::vector<std::string> & forms,
    std::vector<int> & chartypes);

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

      // store the legal transform
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

  // legal y->x
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
