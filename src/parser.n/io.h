#ifndef __LTP_PARSER_CONLL_READER_H__
#define __LTP_PARSER_CONLL_READER_H__

#include <iostream>
#include <fstream>
#include "framework/io.h"
#include "parser.n/instance.h"
#include "parser.n/options.h"

namespace ltp {
namespace depparser {

class CoNLLReader: public framework::LineCountsReader {
private:
  bool trace;
public:
  CoNLLReader(std::istream& _f, bool _trace);

  Instance* next();
};  // end for ConllReader

class CoNLLWriter {
public:
  CoNLLWriter(std::ostream& _f): f(_f) {}

  void write(const Instance& inst);
  void write(const Instance& inst, const std::vector<int>& heads,
      const std::vector<std::string>& deprels);
private:
  std::ostream& f;
};  // end for ConnllWriter

}   // end for parser
}   // end for namespace ltp

#endif  // end for __LTP_PARSER_CONLL_READER_H__
