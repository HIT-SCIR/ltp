#ifndef __LTP_FRAMEWORK_PARAMETER_H__
#define __LTP_FRAMEWORK_PARAMETER_H__

#include <iostream>
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

namespace ltp {
namespace framework {

class Parameters {
public:
  int _dim;
  double * _W;
  double * _W_sum;
  int *    _W_time;

  Parameters()
    : _dim(0),
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

  void add(int idx, int now, const double& scale = 1.) {
    int elapsed = now - _W_time[idx];
    double upd = scale;
    double cur_val = _W[idx];

    _W[idx]       = cur_val + upd;
    _W_sum[idx]   += elapsed * cur_val + upd;
    _W_time[idx]  = now;
  }


  void add(const math::SparseVec& vec, int now, const double& scale = 1.) {
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end();
        ++ itx) {
      int idx = itx->first;
      int elapsed = now - _W_time[idx];
      double upd = scale * itx->second;
      double cur_val = _W[idx];

      _W[idx]     = cur_val + upd;
      _W_sum[idx]  += elapsed * cur_val + upd;
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

  double dot(const int idx, bool use_avg = false) const {
    const double * const p = (use_avg ? _W_sum : _W);
    return p[idx];
  }

  void str(std::ostream& out, int width = 10) {
    out << "\t";
    for (int i = 0; i < width; ++ i) {
      out << "[" << i << "]\t";
    }
    out << std::endl;
    for (int i = 0; i < _dim; ++ i) {
      if (i % width == 0) {
        out << "[" << i << "-" << (i / width + 1)  * width - 1 << "]\t";
      }
      out << _W[i] << "\t";
      if ((i + 1) % width == 0) {
        out << std::endl;
      }
    }
    out << std::endl;
  }

  void flush(int now) {
    for(int i = 0; i < _dim; ++i) {
      _W_sum[i] += (now - _W_time[i]) * _W[i];
      _W_time[i] = now;
    }
  }

  void dump(std::ostream & out, bool use_avg = true) const {
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

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_PARAMETER_H__
