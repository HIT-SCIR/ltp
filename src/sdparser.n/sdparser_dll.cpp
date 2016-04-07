#include "sdparser.n/sdparser_dll.h"
#include "sdparser.n/parser.h"
#include "utils/logging.hpp"
#include "utils/sbcdbc.hpp"
#include <iostream>

class __ltp_dll_sdparser_wrapper : public ltp::sdparser::NeuralNetworkParser {
public:
  __ltp_dll_sdparser_wrapper() {}
  ~__ltp_dll_sdparser_wrapper() {}

  bool load(const char * model_file) {
    if (!ltp::sdparser::NeuralNetworkParser::load(std::string(model_file))) {
      return false;
    }
    setup_system();
    build_feature_space();
    return true;
  }

  int parse(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            std::vector<int> & heads,
            std::vector<std::string> & deprels) {

    ltp::sdparser::Instance inst;
    inst.forms.push_back( ltp::sdparser::SpecialOption::ROOT );
    inst.postags.push_back( ltp::sdparser::SpecialOption::ROOT );

    for (size_t i = 0; i < words.size(); ++ i) {
      inst.forms.push_back(ltp::strutils::chartypes::sbc2dbc(words[i]));
      inst.postags.push_back(postags[i]);
    }

    ltp::sdparser::NeuralNetworkParser::predict(inst, heads, deprels);
    heads.erase(heads.begin());
    deprels.erase(deprels.begin());
    return heads.size();
  }
};

void * sdparser_create_parser(const char * path) {
  __ltp_dll_sdparser_wrapper* wrapper = new __ltp_dll_sdparser_wrapper();

  if (!wrapper->load(path)) {
    delete wrapper;
    return 0;
  }
  return reinterpret_cast<void *>(wrapper);
}

int sdparser_release_parser(void * parser) {
  if (!parser) {
    return -1;
  }
  delete reinterpret_cast<__ltp_dll_sdparser_wrapper*>(parser);
  return 0;
}

int sdparser_parse(void * parser,
                 const std::vector<std::string> & words,
                 const std::vector<std::string> & postags,
                 std::vector<int> & heads,
                 std::vector<std::string> & deprels) {
  if (words.size() != postags.size()) {
    return 0;
  }
  for (int i = 0; i < words.size(); ++ i) {
    if (words[i].empty() || postags[i].empty()) {
      return 0;
    }
  }

  __ltp_dll_sdparser_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_sdparser_wrapper*>(parser);
  return wrapper->parse(words, postags, heads, deprels);
}
