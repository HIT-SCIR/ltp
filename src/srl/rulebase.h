#ifndef __LTP_SRL_RULE_BASED_H__
#define __LTP_SRL_RULE_BASED_H__

#include <vector>
#include <iostream>

namespace ltp {
namespace srl {
namespace rulebase {

static bool dll_validity_check(
    const std::vector<std::string> & words,
    const std::vector<std::string> & postags,
    const std::vector<std::string> & nes,
    const std::vector<std::pair<int,std::string> > & parser) {
  if(words.size()!=postags.size()||words.size()!=parser.size()||words.size()!=nes.size()) {
    return false;
  }
  int len = parser.size(); 
  for(int i = 0;i<len; ++i) {
    int father = parser[i].first;
    if(father<0||father>=len) {
      return false;
    }
  }
  return true;
}

}//end rulebase
}//end postagger
}//end ltp
#endif
