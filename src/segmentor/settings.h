#ifndef __LTP_SEGMENTOR_SETTINGS_H__
#define __LTP_SEGMENTOR_SETTINGS_H__

#include <iostream>

namespace ltp {
namespace segmentor {

const std::string BOS = "_bos_";
const std::string EOS = "_eos_";
const std::string BOT = "_bot_";
const std::string EOT = "_eot_";

const std::string __dummy__     = "dummy";
const std::string __b__         = "b";
const std::string __i__         = "i";
const std::string __e__         = "e";
const std::string __s__         = "s";

const int __b_id__ = 0;
const int __i_id__ = 1;
const int __e_id__ = 2;
const int __s_id__ = 3;

const std::string __eng__       = "_eng_";
const std::string __uri__       = "_uri_";

// Used when input data is partially annotated.
const std::string __partial_start__ = "<ltp:partial>";
const std::string __partial_end__   = "</ltp:partial>";
const std::string __word_start__    = "<ltp:word>";
const std::string __word_end__      = "</ltp:word>";

const int __num_pos_types__ = 4;
static const char* __pos_types__[] = { "b", "i", "e", "s", };

}   //  end for namespace segmentor
}   //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SETTINGS_H__
