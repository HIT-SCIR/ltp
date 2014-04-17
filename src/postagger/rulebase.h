#ifndef __LTP_POSTAGGER_RULE_BASED_H__
#define __LTP_POSTAGGER_RULE_BASED_H__

#include <iostream>
#include <bitset>
#include <vector>

namespace ltp {
namespace postagger {
namespace rulebase {

static bool dll_validity_check(const std::vector<std::string> & words) {
    for (int i = 0; i < words.size(); ++ i) {
        if(0 == words[i].length()) {
            return false;
        }
    }
  return true;
}

}//end rulebase
}//end postagger
}//end ltp
#endif
