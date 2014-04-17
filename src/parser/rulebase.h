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

  int len = words.size();

  for(int i =0;i<len;i++) {
    if(words[i].empty()) {
      return false;
    }
  }

  for(int i =0;i<len;i++) {
    if(postags[i].empty()) {
      return false;
    }
  }

  return true;
}

}//end rulebase
}//end postagger
}//end ltp
#endif
