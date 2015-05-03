#ifndef __LTP_POSTAGGER_DECODER_H__
#define __LTP_POSTAGGER_DECODER_H__

#include <iostream>
#include <map>
#include <vector>
#include "framework/decoder.h"
#include "postagger/instance.h"
#include "postagger/score_matrix.h"
#include "utils/math/mat.h"

namespace ltp {
namespace postagger {

class Decoder: public framework::ViterbiDecoder {
private:
  typedef framework::ViterbiLatticeItem LatticeItem;
  int L;
public:
  Decoder (int _l) : L(_l) {}
  void decode(Instance * inst, const ScoreMatrix* scm);
private:
  void viterbi_decode_inner(const Instance * inst,const ScoreMatrix* scm, int i,int l);
  void viterbi_decode(const Instance * inst, const ScoreMatrix* scm);
  void get_result(Instance * inst);
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_DECODER_H__
