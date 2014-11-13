#ifndef __LTP_SEGMENTOR_DECODER_H__
#define __LTP_SEGMENTOR_DECODER_H__

#include <iostream>
#include <vector>
#include "segmentor/instance.h"
#include "segmentor/rulebase.h"
#include "segmentor/score_matrix.h"
#include "utils/math/mat.h"

namespace ltp {
namespace segmentor {

// data structure for lattice item
class LatticeItem {
public:
  LatticeItem (int _i, int _l, double _score, const LatticeItem * _prev) :
    i(_i),
    l(_l),
    score(_score),
    prev(_prev) {}

  LatticeItem (int _l, double _score) :
    i(0),
    l(_l),
    score(_score),
    prev(0) {}

public:
  int         i;
  int         l;
  double        score;
  const LatticeItem * prev;
};

class Decoder {
public:
  Decoder (int _l, rulebase::RuleBase & _base) : L(_l), base(_base) {}

  //!
  void decode(Instance * inst, const ScoreMatrix* scm);

private:
  void init_lattice(const Instance * inst);
  void viterbi_decode(const Instance * inst, const ScoreMatrix* scm);
  void get_result(Instance * inst);
  void free_lattice();

private:
  int L;

  math::Mat< const LatticeItem * > lattice;
  rulebase::RuleBase base;

  void lattice_insert(const LatticeItem * &position,
                      const LatticeItem * const item) {
    if (position == NULL) {
      position = item;
    } else if (position->score < item->score) {
      delete position;
      position = item;
    } else {
      delete item;
    }
  }
};

}       //  end for namespace segmentor
}       //  end for namespace ltp
#endif    //  end for __LTP_SEGMENTOR_DECODER_H__
