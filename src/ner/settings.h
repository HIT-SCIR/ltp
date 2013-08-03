#ifndef __LTP_NER_SETTINGS_H__
#define __LTP_NER_SETTINGS_H__

#include <iostream>

namespace ltp {
namespace ner {

const std::string BOS = "_bos_";
const std::string EOS = "_eos_";
const std::string BOP = "_bop_";
const std::string EOP = "_eop_";

const double EPS        = 1e-8;
const double NEG_INF    = -1e20;

const int __num_pos_types__ = 4;
const int __num_ne_types__  = 3;

static const char * __pos_types__[]    = { "B", "I", "E", "S", };
static const char * __ne_types__[]     = { "Nh", "Ni", "Ns", };

}   //  end for namespace ner
}   //  end for namespace ltp

#endif  //  end for __LTP_NER_SETTINGS_H__
