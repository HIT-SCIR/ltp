#ifndef __LTP_NER_DECODER_H__
#define __LTP_NER_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "ner/instance.h"
#include "ner/rulebase.h"
#include "utils/math/mat.h"

namespace ltp {
namespace ner {

// data structure for lattice item
class Decoder: public framework::ViterbiDecoder {
private:
  typedef framework::ViterbiLatticeItem LatticeItem;
  int L;
  rulebase::RuleBase base;

public:
  Decoder (int _l, rulebase::RuleBase& _base) : L(_l), base(_base) {}
  void decode(Instance * inst);

private:
  void viterbi_decode(const Instance * inst);
  void get_result(Instance * inst);
};

}           //  end for namespace ner
}           //  end for namespace ltp
#endif      //  end for __LTP_NER_DECODER_H__
