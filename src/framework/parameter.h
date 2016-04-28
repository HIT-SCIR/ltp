#ifndef __LTP_FRAMEWORK_PARAMETER_H__
#define __LTP_FRAMEWORK_PARAMETER_H__

#include <iostream>
#include <cstring>
#include "boost/cstdint.hpp"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"
#include "utils/logging.hpp"

namespace ltp {
namespace framework {

using boost::uint32_t;

class Parameters {
public:
  bool _enable_wrapper;

  uint32_t _dim;
  uint32_t _last_timestamp;

  double* _W;
  double* _W_sum;
  uint32_t* _W_time;

public:
  enum DumpOption {
    kDumpAveraged = 0,
    kDumpNonAveraged,
    kDumpDetails
  };

  Parameters()
    : _dim(0), _W(0), _W_sum(0), _W_time(0),
    _last_timestamp(0), _enable_wrapper(false) {}
  ~Parameters() { dealloc(); }

  void realloc(const uint32_t& dim) {
    dealloc();
    _dim = dim;

    if (dim > 0) {
      _W = new double[dim];
      _W_sum = new double[dim];
      _W_time = new uint32_t[dim];
    }

    for (uint32_t i = 0; i < dim; ++ i) {
      _W[i] = 0;
      _W_sum[i] = 0;
      _W_time[i] = 0;
    }
  }

  void dealloc() {
    if (_W && _W == _W_sum) {
      // _W_sum is loaded as a wrapper. Then only one time free is fine.
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

  bool is_wrapper() const { return _enable_wrapper; }
  size_t last() const { return _last_timestamp; }

  /**
   * Increase one dimension in the parameter vector by scale.
   *
   *  @param[in]  idx   The index to this dimension.
   *  @param[in]  now   The timestamp.
   *  @param[in]  scale The scale
   */
  void add(const uint32_t& idx, const uint32_t& now, const double& scale = 1.) {
    uint32_t elapsed = now - _W_time[idx];
    double cur_val = _W[idx];

    _W[idx] = cur_val + scale;
    _W_sum[idx] += elapsed * cur_val + scale;
    _W_time[idx] = now;

    if (_last_timestamp < now) {
      _last_timestamp = now;
    }
  }

  /**
   * Increase several dimensions in the parameter vector by scale.
   *
   *  @param[in]  vec   Sparse vector of indices.
   *  @param[in]  now   The timestamp.
   *  @param[in]  scale The scale
   */
  void add(const math::SparseVec& vec, const uint32_t& now, const double& scale = 1.) {
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end(); ++ itx) {
      uint32_t idx = itx->first;
      uint32_t elapsed = now - _W_time[idx];
      double upd = scale * itx->second;
      double cur_val = _W[idx];

      _W[idx] = cur_val + upd;
      _W_sum[idx] += elapsed * cur_val + upd;
      _W_time[idx] = now;
    }

    if (_last_timestamp < now) {
      _last_timestamp = now;
    }
  }

  /**
   * Get the dot product of the parameter vector with a sparse vector.
   *
   *  @param[in]  vec     Sparse vector of indices.
   *  @param[in]  avg     If true, use the averaged parameter (_W_sum), otherwise use the
   *                      non-averaged one (_W).
   *  @return     double  The dot product.
   */
  double dot(const math::SparseVec& vec, bool avg = false) const {
    const double * const p = (avg ? _W_sum : _W);
    double ret = 0.;
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end();
        ++ itx) {
      ret += p[itx->first] * itx->second;
    }
    return ret;
  }

  /**
   * Get the dot product of the parameter vector with a feature vector.
   *
   *  @param[in]  vec     The feature vector (coupled with labels).
   *  @param[in]  avg     If true, use the averaged parameter (_W_sum), otherwise use the
   *                      non-averaged one (_W).
   *  @return     double  The dot product.
   */
  double dot(const math::FeatureVector* vec, bool avg = false) const {
    const double * const p = (avg ? _W_sum : _W);
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

  /**
   * Get the weight of a single dimension.
   *
   *  @param[in]  vec     The feature vector (coupled with labels).
   *  @param[in]  avg     If true, use the averaged parameter (_W_sum), otherwise use the
   *                      non-averaged one (_W).
   *  @return     double  The dot product.
   */
  double dot(const uint32_t idx, bool avg = false) const {
    const double * const p = (avg ? _W_sum : _W);
    return p[idx];
  }

  double predict(const math::FeatureVector* vec, const uint32_t& elapsed_time) const {
    double ret = 0;
    for (uint32_t i = 0; i < vec->n; ++i) {
      uint32_t idx = vec->idx[i] + vec->loff;
      if (vec->val) {
        ret += (_W_sum[idx] + _W[idx] * elapsed_time * vec->val[i]);
      }
      else {
        ret += (_W_sum[idx] + _W[idx] * elapsed_time);
      }
    }
    return ret;
  }

  double predict(const uint32_t idx, const uint32_t& elapsed_time) const {
    return _W_sum[idx] + _W[idx] * elapsed_time;
  }

  /**
   * Flush the parameter vector.
   *
   *  @param[in]  now   The timestamp.
   */
  void flush(const uint32_t& now) {
    for (uint32_t i = 0; i < _dim; ++i) {
      _W_sum[i] += (now - _W_time[i]) * _W[i];
      _W_time[i] = now;
    }

    if (_last_timestamp < now) {
      _last_timestamp = now;
    }
  }

  void str(std::ostream& out, uint32_t width = 10) {
    if (0 == width) return;
    out << "\t";
    for (uint32_t i = 0; i < width; ++ i) {
      out << "[" << i << "]\t";
    }
    out << std::endl;
    for (uint32_t i = 0; i < _dim; ++ i) {
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

  /**
   * Dump the parameter vector to the output stream according to the dump option.
   * If kDumpDetails is configured, it will dump the _W, _W_sum along with the
   * _last_timestamp; If kDumpAveraged is configured, dump the _W_sum; If
   * kDumpNonAveraged is configured, dump _W.
   *
   *  @param[out] out   The output stream.
   *  @param[in]  opt   The dump option.
   */
  void dump(std::ostream & out, const DumpOption& opt) const {
    char chunk[16];
    if (opt == kDumpDetails) {
      strncpy(chunk, "param-details", 16);
    } else if (opt == kDumpAveraged) {
      strncpy(chunk, "param-avg", 16);
    } else if (opt == kDumpNonAveraged) {
      strncpy(chunk, "param-nonavg", 16);
    }
    out.write(chunk, 16);
    out.write(reinterpret_cast<const char*>(&_dim), sizeof(uint32_t));

    if (_dim > 0) {
      if (opt == kDumpDetails) {
        out.write(reinterpret_cast<const char*>(_W), sizeof(double) * _dim);
        out.write(reinterpret_cast<const char*>(_W_sum), sizeof(double) * _dim);
        out.write(reinterpret_cast<const char*>(&_last_timestamp), sizeof(uint32_t));
      } else if (opt == kDumpAveraged) {
        out.write(reinterpret_cast<const char*>(_W_sum), sizeof(double) * _dim);
        out.write(reinterpret_cast<const char*>(&_last_timestamp), sizeof(uint32_t));
      } else if (opt == kDumpNonAveraged) {
        out.write(reinterpret_cast<const char*>(_W), sizeof(double) * _dim);
      }
    }
  }

  /**
   * Load the model from input file stream.
   *
   *  @param[in]  in    The input stream
   *  @return     bool  If the model file is well-formated, return true; otherwise
   *                    false.
   */
  bool load(std::istream & in) {
    char chunk[16];
    in.read(chunk, 16);
    char header[16];
    char body[16];
    strncpy(header, chunk, 5); header[5] = 0;
    strncpy(body, chunk+ 6, 11);
    if (strcmp(header, "param")) {
      return false;
    }

    in.read(reinterpret_cast<char *>(&_dim), sizeof(uint32_t));
    if (_dim > 0) {
      if (!strncmp(body, "details", 11)) {
        _W = new double[_dim];
        _W_sum = new double[_dim];
        in.read(reinterpret_cast<char *>(_W), sizeof(double)* _dim);
        in.read(reinterpret_cast<char *>(_W_sum), sizeof(double)* _dim);
        in.read(reinterpret_cast<char *>(&_last_timestamp), sizeof(uint32_t));
        _enable_wrapper = false;
      } else if (!strncmp(body, "avg", 11)) {
        _W_sum = new double[_dim];
        in.read(reinterpret_cast<char *>(_W_sum), sizeof(double)* _dim);
        in.read(reinterpret_cast<char *>(&_last_timestamp), sizeof(uint32_t));
        _W = _W_sum;
        _enable_wrapper = true;
      } else if (!strncmp(body, "nonavg", 11)) {
        _W = new double[_dim];
        in.read(reinterpret_cast<char *>(_W), sizeof(double)* _dim);
        _W_sum = _W;
        _enable_wrapper = true;
      } else {
        WARNING_LOG("model dump method is not specified!");
      }
    }
    return true;
  }
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_PARAMETER_H__
