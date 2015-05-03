#ifndef __LTP_FRAMEWORK_DECODER_H__
#define __LTP_FRAMEWORK_DECODER_H__

#include "utils/math/mat.h"

namespace ltp {
namespace framework {

class ViterbiLatticeItem {
public:
  ViterbiLatticeItem (int _i, int _l, const double& _score,
      const ViterbiLatticeItem * _prev) :
    i(_i),
    l(_l),
    score(_score),
    prev(_prev) {}

  ViterbiLatticeItem (int _l, const double& _score) :
    i(0),
    l(_l),
    score(_score),
    prev(0) {}

public:
  int i;
  int l;
  double score;
  const ViterbiLatticeItem * prev;
};

class ViterbiDecoder {
protected:
  void init_lattice(int len, int nr_labels) {
    lattice.resize(len, nr_labels);
    lattice = NULL;
  }

  void free_lattice() {
    int len = lattice.total_size();
    const ViterbiLatticeItem ** p = lattice.c_buf();
    for (int i = 0; i < len; ++ i) {
      if (p[i]) { delete p[i]; }
    }
  }

  void lattice_insert(const ViterbiLatticeItem * &position,
      const ViterbiLatticeItem * const item) {
    if (position == NULL) {
      position = item;
    } else if (position->score < item->score) {
      delete position;
      position = item;
    } else {
      delete item;
    }
  }

  math::Mat< const ViterbiLatticeItem * > lattice;
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_DECODER_H__
