#ifndef __LTP_SEGMENTOR_DECODER_H__
#define __LTP_SEGMENTOR_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "segmentor/instance.h"
#include "segmentor/rulebase.h"
#include "segmentor/score_matrix.h"
#include "utils/math/mat.h"

namespace ltp {
namespace segmentor {

class Decoder: public framework::ViterbiDecoder {
private:
  typedef framework::ViterbiLatticeItem LatticeItem;
  int L;
  rulebase::RuleBase base;
public:
  Decoder (int _l, rulebase::RuleBase & _base) : L(_l), base(_base) {}

  /**
   * The main decoding process
   *  @param[in/out]  the instance
   *  @param[in]  the score matrix
   */
  void decode(Instance * inst, const ScoreMatrix* scm);

private:
  void viterbi_decode(const Instance * inst, const ScoreMatrix* scm);
  void get_result(Instance * inst);
};

}       //  end for namespace segmentor
}       //  end for namespace ltp
#endif    //  end for __LTP_SEGMENTOR_DECODER_H__
