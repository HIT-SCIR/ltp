#ifndef __LTP_POSTAGGER_DECODER_H__
#define __LTP_POSTAGGER_DECODER_H__

#include <iostream>
#include <map>
#include <vector>
#include "postagger/instance.h"
#include "utils/math/mat.h"

namespace ltp {
namespace postagger {

// data structure for lattice item
class LatticeItem {
public:
  LatticeItem (int _i, int _l, double _score, const LatticeItem * _prev) 
    : i(_i),
      l(_l),
      score(_score),
      prev(_prev) {}

  LatticeItem (int _l, double _score) 
    : i(0),
      l(_l),
      score(_score),
      prev(0) {}

public:
  int         i;
  int         l;
  double      score;
  const LatticeItem * prev;
};

class Decoder {
public:
  Decoder (int _L) : L(_L) {}
  void decode(Instance * inst);
private:
  void init_lattice(const Instance * inst);
  void viterbi_decode_inner(const Instance * inst,int i,int l);
  void viterbi_decode(const Instance * inst);
  void get_result(Instance * inst);
  void free_lattice();

private:
  int L;

  math::Mat< const LatticeItem * > lattice;

  void lattice_insert(const LatticeItem * &position, const LatticeItem * const item) {
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

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_DECODER_H__
