#ifndef __LTP_SEGMENTOR_DECODE_CONTEXT_H__
#define __LTP_SEGMENTOR_DECODE_CONTEXT_H__

namespace ltp {
namespace segmentor {

class DecodeContext {
public:
  //! the gold features.
  math::SparseVec correct_features;
  //! the predicted features.
  math::SparseVec predicted_features;
  //!
  math::SparseVec updated_features;

  //! The feature cache.
  math::Mat< math::FeatureVector *> uni_features;

  DecodeContext() {}
  ~DecodeContext() {}

  void clear() {
    if (uni_features.total_size() > 0) {
      int d1 = uni_features.nrows();
      int d2 = uni_features.ncols();
      for (int i = 0; i < d1; ++ i) {
        if (uni_features[i][0]) {
          uni_features[i][0]->clear();
        }
        for (int j = 0; j < d2; ++ j) {
          if (uni_features[i][j]) {
            delete uni_features[i][j];
          }
        }
      }
    }

    uni_features.dealloc();
    correct_features.zero();
    predicted_features.zero();
  }
};

} //  end for namespace segmentor
} //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_DECODE_CONTEXT_H__
