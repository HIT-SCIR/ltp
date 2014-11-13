#ifndef __LTP_SEGMENTOR_SCORE_MATRIX_H__
#define __LTP_SEGMENTOR_SCORE_MATRIX_H__

#include "utils/math/mat.h"

namespace ltp {
namespace segmentor {

struct ScoreMatrix {
  math::Mat< double > uni_scores;
  math::Mat< double > bi_scores;

  void clear() {
    uni_scores.dealloc();
    bi_scores.dealloc();
  }
};

} //  end for namespace segmentor
} //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SCORE_MATRIX_H__
