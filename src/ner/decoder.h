#ifndef __LTP_NER_DECODER_H__
#define __LTP_NER_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "utils/smartmap.hpp"
#include "utils/unordered_set.hpp"

namespace ltp {
namespace ner {

class NERTransitionConstrain: public framework::ViterbiDecodeConstrain {
private:
  std::unordered_set<size_t> rep;
  size_t T;
public:
  NERTransitionConstrain(const utility::IndexableSmartMap& labels,
      const std::vector<std::string>& includes);

  bool can_tran(const size_t& i, const size_t& j) const;
};

}           //  end for namespace ner
}           //  end for namespace ltp
#endif      //  end for __LTP_NER_DECODER_H__
