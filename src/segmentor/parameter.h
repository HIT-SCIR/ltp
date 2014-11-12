#ifndef __LTP_SEGMENTOR_PARAMETER_H__
#define __LTP_SEGMENTOR_PARAMETER_H__

#include <iostream>
#include <cstring>
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

#define SEGMENTOR_PARAM         "param"         // for model version lower than 3.2.0
#define SEGMENTOR_PARAM_FULL    "param-full"
#define SEGMENTOR_PARAM_MINIMAL "param-minimal"

namespace ltp {
namespace segmentor {

class Parameters {
public:
  int _dim;
  double * _W;
  double * _W_sum;
  int *    _W_time;

  Parameters() :
    _dim(0),
    _W(0),
    _W_sum(0),
    _W_time(0) {}

  ~Parameters() {
    dealloc();
  }

  void realloc(int dim) {
    dealloc();
    _dim = dim;

    if (dim > 0) {
      _W = new double[dim];
      _W_sum = new double[dim];
      _W_time = new int[dim];
    }

    for (int i = 0; i < dim; ++ i) {
      _W[i] = 0;
      _W_sum[i] = 0;
      _W_time[i] = 0;
    }
  }

  void dealloc() {
    if (_W && _W == _W_sum) {
      delete [](_W);
      _W = 0;
      _W_sum = 0;
    } else {
      if (_W) {
        delete [](_W);
        _W = 0;
      }
      if (_W_sum) {
        delete [](_W_sum);
        _W_sum = 0;
      }
    }

    if (_W_time) {
      delete [](_W_time);
      _W_time = 0;
    }
  }

  void add(int idx, int now, double scale = 1.) {
    int elapsed = now - _W_time[idx];
    double upd = scale;
    double cur_val = _W[idx];

    _W[idx]       = cur_val + upd;
    _W_sum[idx]   += elapsed * cur_val + upd;
    _W_time[idx]  = now;
  }

  void add(const math::SparseVec & vec, int now, double scale = 1.) {
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end();
        ++ itx) {
      int idx = itx->first;
      int elapsed = now - _W_time[idx];
      double upd = scale * itx->second;
      double cur_val = _W[idx];

      _W[idx]       = cur_val + upd;
      _W_sum[idx]   += elapsed * cur_val + upd;
      _W_time[idx]  = now;
    }
  }

  double dot(const math::SparseVec & vec, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    double ret = 0.;
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end();
        ++ itx) {
      ret += p[itx->first] * itx->second;
    }
    return ret;
  }

  double dot(const math::FeatureVector * vec, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    double ret = 0.;
    for (int i = 0; i < vec->n; ++ i) {
      if (vec->val) {
        ret += p[vec->idx[i] + vec->loff] * vec->val[i];
      } else {
        ret += p[vec->idx[i] + vec->loff];
      }
    }
    return ret;
  }

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

  double dot(const int idx, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    return p[idx];
  }

  double dot_flush_time(const int idx, int beg_time, int end_time) const {
    return _W_sum[idx] + _W[idx] * (end_time - beg_time);
  }

  void flush(int now) {
    for(int i = 0; i < _dim; ++i) {
      _W_sum[i]  += (now - _W_time[i]) * _W[i];
      _W_time[i] = now;
    }
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

