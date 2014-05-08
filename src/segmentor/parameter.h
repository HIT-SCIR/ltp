#ifndef __LTP_SEGMENTOR_PARAMETER_H__
#define __LTP_SEGMENTOR_PARAMETER_H__

#include <iostream>
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

namespace ltp {
namespace segmentor {

using namespace ltp::math;

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

  void add(const SparseVec & vec, int now, double scale = 1.) {
    for (SparseVec::const_iterator itx = vec.begin();
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

  double dot(const SparseVec & vec, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    double ret = 0.;
    for (SparseVec::const_iterator itx = vec.begin();
        itx != vec.end();
        ++ itx) {
      ret += p[itx->first] * itx->second;
    }
    return ret;
  }

  double dot(const FeatureVector * vec, bool use_avg = false) const {
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

  double dot(const int idx, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    return p[idx];
  }

  void flush(int now) {
    for(int i = 0; i < _dim; ++i) {
      _W_sum[i]  += (now - _W_time[i]) * _W[i];
      _W_time[i] = now;
    }
  }

  void dump(std::ostream & out, bool use_avg = true) {
    const double * p = (use_avg ? _W_sum : _W);
    char chunk[16] = {'p', 'a', 'r', 'a', 'm', 0};
    out.write(chunk, 16);
    out.write(reinterpret_cast<const char *>(&_dim), sizeof(int));
    if (_dim > 0) {
      out.write(reinterpret_cast<const char *>(p), sizeof(double) * _dim);
    }
  }

  bool load(std::istream & in) {
    char chunk[16];
    in.read(chunk, 16);
    if (strcmp(chunk, "param")) {
      return false;
    }

    in.read(reinterpret_cast<char *>(&_dim), sizeof(int));
    if (_dim > 0) {
      _W = new double[_dim];
      in.read(reinterpret_cast<char *>(_W), sizeof(double) * _dim);
      _W_sum = _W;
    }

    return true;
  }
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_PARAMETER_H__

