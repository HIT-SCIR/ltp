#ifndef __LTP_POSTAGGER_SETTINGS_H__
#define __LTP_POSTAGGER_SETTINGS_H__

#include <iostream>

namespace ltp {
namespace postagger {

const std::string BOS = "_bos_";
const std::string EOS = "_eos_";
const std::string BOT = "_bot_";
const std::string EOT = "_eot_";
const std::string BOC = "_boc_";
const std::string EOC = "_eoc_";

const double EPS        = 1e-8;
const double NEG_INF    = -1e20;

}   //  end for namespace postagger
}   //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_SETTINGS_H__
