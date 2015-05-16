#ifndef __LTP_NER_SETTINGS_H__
#define __LTP_NER_SETTINGS_H__

#include <iostream>

namespace ltp {
namespace ner {

const std::string BOS = "_bos_";
const std::string EOS = "_eos_";
const std::string BOP = "_bop_";
const std::string EOP = "_eop_";
const std::string OTHER = "O";

const int __num_pos_types__ = 4;
static const char* __pos_types__[] = { "B", "I", "E", "S", };

}   //  end for namespace ner
}   //  end for namespace ltp

#endif  //  end for __LTP_NER_SETTINGS_H__
