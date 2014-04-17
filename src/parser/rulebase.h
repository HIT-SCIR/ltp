#ifndef __LTP_PARSER_RULE_BASED_H__
#define __LTP_PARSER_RULE_BASED_H__

#include <vector>

namespace ltp {
namespace parser {
namespace rulebase {

static bool dll_validity_check(const std::vector<std::string> & words,const std::vector<std::string> & postags) {
  if(words.size()!=postags.size()) {
    return false;
  }
  return true;
}

}//end rulebase
}//end postagger
}//end ltp
#endif
