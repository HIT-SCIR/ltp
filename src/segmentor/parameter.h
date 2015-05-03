#ifndef __LTP_SEGMENTOR_PARAMETER_H__
#define __LTP_SEGMENTOR_PARAMETER_H__

#include <iostream>
#include <cstring>
#include "framework/parameter.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

#define SEGMENTOR_PARAM         "param"         // for model version lower than 3.2.0
#define SEGMENTOR_PARAM_FULL    "param-full"
#define SEGMENTOR_PARAM_MINIMAL "param-minimal"

namespace ltp {
namespace segmentor {

class Parameters: public framework::Parameters {
public:
  Parameters() {}
  ~Parameters() {}

  double dot_flush_time(const math::FeatureVector * vec, int beg_time, int end_time) const {
    double ret = 0;
    for (int i = 0; i < vec->n; ++ i) {
      int idx = vec->idx[i] + vec->loff;
      if (vec->val) {
        ret += (_W_sum[idx] + _W[idx] * (end_time - beg_time) * vec->val[i]);
      } else {
        ret += (_W_sum[idx] + _W[idx] * (end_time - beg_time));
      }
    }
    return ret;
  }

  double dot_flush_time(const int idx, int beg_time, int end_time) const {
    return _W_sum[idx] + _W[idx] * (end_time - beg_time);
  }

  //! Dump the model. since version 3.2.0, fully dumped model is supported.
  //! using a tag full to distinguish between old and new model.
  void dump(std::ostream & out, bool full) {
    char chunk[16];

    if (full) {
      strncpy(chunk, SEGMENTOR_PARAM_FULL,16);
    } else {
      strncpy(chunk, SEGMENTOR_PARAM_MINIMAL, 16);
    }

    out.write(chunk, 16);
    out.write(reinterpret_cast<const char *>(&_dim), sizeof(int));
    if (_dim > 0) {
      if (full) {
        out.write(reinterpret_cast<const char *>(_W), sizeof(double) * _dim);
      }
      out.write(reinterpret_cast<const char *>(_W_sum), sizeof(double) * _dim);
    }
  }

  bool load(std::istream & in, bool full) {
    char chunk[16];

    in.read(chunk, 16);
    if (!(
          (!strcmp(chunk, SEGMENTOR_PARAM_FULL) && full) ||
          (!strcmp(chunk, SEGMENTOR_PARAM_MINIMAL) && !full) ||
          (!strcmp(chunk, SEGMENTOR_PARAM) && !full)
          )) {
      return false;
    }

    in.read(reinterpret_cast<char *>(&_dim), sizeof(int));

    if (_dim > 0) {
      if (full) {
        _W = new double[_dim];
        _W_sum = new double[_dim];
        in.read(reinterpret_cast<char *>(_W), sizeof(double) * _dim);
        in.read(reinterpret_cast<char *>(_W_sum), sizeof(double) * _dim);
      } else {
        _W = new double[_dim];
        in.read(reinterpret_cast<char *>(_W), sizeof(double) * _dim);
        _W_sum = _W;
      }
    }

    return true;
  }
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_PARAMETER_H__

