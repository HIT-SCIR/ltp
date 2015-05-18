#ifndef __LTP_FEATURE_VECTOR_H__
#define __LTP_FEATURE_VECTOR_H__

namespace ltp {
namespace math {

struct FeatureVector {
public:
  FeatureVector () : n(0), idx(0), val(0), loff(0) {
  }

  ~FeatureVector() {
  }

  void clear() {
    if (idx) {
      delete [](idx);
      idx = 0;
    }

    if (val) {
      delete [](val);
      val = 0;
    }
  }

public:
  size_t n;
  int* idx;
  double* val;
  size_t loff;
};

}     //  end for namespace math
}     //  end for namespace ltp

#endif  //  end for __LTP_FEATRUE_VECTOR_H__
