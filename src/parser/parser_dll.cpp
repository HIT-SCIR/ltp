#include "parser/parser_dll.h"
#include "parser/parser.h"
#include "parser/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

class ParserWrapper : public ltp::parser::Parser {
public:
  ParserWrapper() {}
  ~ParserWrapper() {}

  bool load(const char * model_file) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::parser::Model;
    if (!model->load(mfs)) {
      delete model;
      return false;
    }

//    ltp::parser::Parser::build_decoder();

    return true;
  }

  int parse(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            std::vector<int> & heads,
            std::vector<std::string> & deprels) {

    ltp::parser::Instance * inst = new ltp::parser::Instance;
    inst->forms.push_back( ltp::parser::ROOT_FORM );
    inst->postags.push_back( ltp::parser::ROOT_POSTAG );

    for (int i = 0; i < words.size(); ++ i) {
      inst->forms.push_back(words[i]);
      inst->postags.push_back(postags[i]);
    }

    ltp::parser::Parser::extract_features(inst);
    ltp::parser::Parser::calculate_score(inst, ltp::parser::Parser::model->param);

    ltp::parser::Decoder * deco;
    deco = build_decoder();
    deco->decode(inst);

    int len = inst->size();
    heads.resize(len - 1);
    deprels.resize(len - 1);
    for (int i = 1; i < len; ++ i) {
      heads[i - 1] = inst->predicted_heads[i];
      deprels[i - 1] = ltp::parser::Parser::model->deprels.at(
          inst->predicted_deprelsidx[i]);
    }

    delete inst;
    delete deco;

    return heads.size();
  }
};

void * parser_create_parser(const char * path) {
  ParserWrapper * wrapper = new ParserWrapper();

  if (!wrapper->load(path)) {
    return 0;
  }
  return reinterpret_cast<void *>(wrapper);
}

int parser_release_parser(void * parser) {
  if (!parser) {
    return -1;
  }
  delete reinterpret_cast<ParserWrapper *>(parser);
  return 0;
}

int parser_parse(void * parser,
                 const std::vector<std::string> & words,
                 const std::vector<std::string> & postags,
                 std::vector<int> & heads,
                 std::vector<std::string> & deprels) {
  // std::cout << "input str = " << str << std::endl;
  if(!ltp::parser::rulebase::dll_validity_check(words,postags)) {
    return -1;
  }
  ParserWrapper * wrapper = 0;
  wrapper = reinterpret_cast<ParserWrapper *>(parser);
  return wrapper->parse(words, postags, heads, deprels);
}
