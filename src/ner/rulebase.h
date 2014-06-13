#ifndef __LTP_NER_RULE_BASED_H__
#define __LTP_NER_RULE_BASED_H__

#include <iostream>
#include <sstream>
#include <bitset>
#include <vector>

#include "ner/settings.h"
#include "utils/sbcdbc.hpp"
#include "utils/smartmap.hpp"
#include "utils/chartypes.hpp"

namespace ltp {
namespace ner {
namespace rulebase {

class RuleBase {
public:
  RuleBase(utility::IndexableSmartMap & labels) {
    // only 4 tag style is supported

    std::stringstream S;

    __trans__ = 0;
    // b
    S.str(std::string()); S << __pos_types__[0] << "-" << __ne_types__[0];
    __b_idx__ = prefix( labels.index(S.str()) ) ;

    S.str(std::string()); S << __pos_types__[1] << "-" << __ne_types__[0];
    __i_idx__ = prefix( labels.index(S.str()) );

    S.str(std::string()); S << __pos_types__[2] << "-" << __ne_types__[0];
    __e_idx__ = prefix( labels.index(S.str()) );

    S.str(std::string()); S << __pos_types__[3] << "-" << __ne_types__[0];
    __s_idx__ = prefix( labels.index(S.str()) );
    __o_idx__ = prefix( labels.index("O") );

    if (__s_idx__>=0 && __b_idx__>=0 && __i_idx__>=0 && __e_idx__>=0  && __o_idx__>=0) {
      __trans__ |= (1<<((__s_idx__<<3) + __s_idx__));
      __trans__ |= (1<<((__s_idx__<<3) + __b_idx__));
      __trans__ |= (1<<((__s_idx__<<3) + __o_idx__));

      __trans__ |= (1<<((__b_idx__<<3) + __i_idx__));
      __trans__ |= (1<<((__b_idx__<<3) + __e_idx__));

      __trans__ |= (1<<((__i_idx__<<3) + __i_idx__));
      __trans__ |= (1<<((__i_idx__<<3) + __e_idx__));

      __trans__ |= (1<<((__e_idx__<<3) + __s_idx__));
      __trans__ |= (1<<((__e_idx__<<3) + __b_idx__));
      __trans__ |= (1<<((__e_idx__<<3) + __o_idx__));

      __trans__ |= (1<<((__o_idx__<<3) + __s_idx__));
      __trans__ |= (1<<((__o_idx__<<3) + __b_idx__));
      __trans__ |= (1<<((__o_idx__<<3) + __o_idx__));
    } else {
      __trans__ = 0xffff;
    }
  }

  ~RuleBase() {
  }

  inline bool legal_trans(int prev, int curr) {
    int prev_prefix = prefix(prev);
    int prev_suffix = suffix(prev);
    int curr_prefix = prefix(curr);
    int curr_suffix = suffix(curr);

    if (prev_prefix == __b_idx__ || prev_prefix == __i_idx__) {
      return ((__trans__ & (1<<((prev_prefix<<3) + curr_prefix))) > 0
        && (prev_suffix == curr_suffix));
    } else {
      return ((__trans__ & (1<<((prev_prefix<<3) + curr_prefix))) > 0);
    }
  }

private:
  unsigned __trans__;

  int __s_idx__;
  int __b_idx__;
  int __i_idx__;
  int __e_idx__;
  int __o_idx__;

  inline int prefix(int tag) {
    return (tag / __num_ne_types__);
  }

  inline int suffix(int tag) {
    return (tag % __num_ne_types__);
  }
};

}     //  end for rulebase
}     //  end for namespace ner
}     //  end for namespace ltp 

#endif  //  end for __LTP_NER_RULE_BASE_H__
