#ifndef __LTP_PARSER_CONLL_WRITER_H__
#define __LTP_PARSER_CONLL_WRITER_H__

#include <iostream>

#include "utils/strutils.hpp"
#include "parser/instance.h"

namespace ltp {
namespace parser {

using namespace ltp::strutils;

class CoNLLWriter {
public:
  CoNLLWriter(std::ostream& _f): f(_f)  {}
  ~CoNLLWriter() {}

  void write(const Instance * inst) {
    int len = inst->size();
    bool predicted = (inst->predicted_heads.size() > 0
                      && inst->predicted_heads.size() == len);
    bool predicted_label = (inst->predicted_deprels.size() > 0
                            && inst->predicted_deprels.size() == len);

    for (int i = 1; i < inst->size(); ++ i) {
      f << i
        << "\t"           // 0 - index
        << inst->forms[i]
        << "\t"           // 1 - form
        << inst->lemmas[i]
        << "\t"           // 2 - lemma
        << inst->postags[i]
        << "\t"           // 3 - postag
        << "_"
        << "\t"           // 4 - unknown
        << "_"
        << "\t"           // 5 - unknown
        << inst->heads[i]
        << "\t"           // 6 - heads
        << inst->deprels[i]
        << "\t"           // 7 - deprels
        << (predicted ? to_str(inst->predicted_heads[i]) : "_")
        << "\t"
        << (predicted_label ? inst->predicted_deprels[i] : "_")
        << endl;
    }

    f << endl;
  }
private:
  std::ostream& f;
};  // end for ConnllWriter

}   // end for parser
}   // end for namespace ltp


#endif  // end for __LTP_PARSER_CONLL_WRITER_H__
