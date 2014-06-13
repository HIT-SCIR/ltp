#ifndef __LTP_PARSER_CONLL_READER_H__
#define __LTP_PARSER_CONLL_READER_H__

#include <iostream>
#include <fstream>

#include "utils/codecs.hpp"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"
#include "parser/settings.h"
#include "parser/instance.h"
#include "parser/options.h"

namespace ltp {
namespace parser {

using namespace std;
using namespace ltp::strutils;

class CoNLLReader {
public:
  /*
   * Constructor for ConllReader
   * Register a ifstream to the ConllReader
   *
   *  @param  f   the reference to the ifstream
   */
  CoNLLReader(ifstream& _f): f(_f) {}
  ~CoNLLReader() {}

  /*
   * Get next instance from ifstream buffer
   */
  Instance * next() {
    if (f.eof()) {
      return NULL;
    }

    Instance * inst = new Instance;
    string line;

    inst->forms.push_back( ROOT_FORM );
    inst->lemmas.push_back( ROOT_LEMMA );
    inst->postags.push_back( ROOT_POSTAG );
    inst->heads.push_back( -1 );

    if (model_opt.labeled) {
      inst->deprels.push_back( ROOT_DEPREL );
    }
    inst->chars.push_back( vector<string>() );

    while (!f.eof()) {
      getline(f, line);
      chomp(line);

      if (line.size() == 0) {
        break;
      }

      vector<string> items = split(line);
      if (items.size() != 10) {
        WARNING_LOG("Unknown conll format file");
      }

      inst->forms.push_back( items[1] );    // items[1]: form
      inst->lemmas.push_back( items[2] );   // items[2]: lemma
      inst->postags.push_back( items[3] );  // items[4]: postag
      inst->heads.push_back( to_int(items[6]) );

      if (model_opt.labeled) {
        inst->deprels.push_back( items[7] );
      }

      vector<string> chars;
      codecs::decode(items[1], chars);
      inst->chars.push_back( chars );
    }

    if (inst->forms.size() == 1) {
      delete inst;
      inst = NULL;
    }
    return inst;
  }

  /*
   * Reader reach the end of the file
   */
  bool eof() {
    return f.eof();
  }
private:
  ifstream& f;
};  // end for ConllReader
}   // end for parser
}   // end for namespace ltp

#endif  // end for __LTP_PARSER_CONLL_READER_H__
