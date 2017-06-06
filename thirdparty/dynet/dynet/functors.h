#ifndef DYNET_GPU_FUNCTORS_H
#define DYNET_GPU_FUNCTORS_H

#include <cstdint>
#include <cmath>
#include <limits>

#if HAVE_CUDA
#  define DYNET_DEVICE_FUNC __device__
#  define DYNET_DEVICE_MIN 1.175494351e-38f
#else
#  define DYNET_DEVICE_FUNC
#  define DYNET_DEVICE_MIN std::numeric_limits<float>::min()
#endif

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it
//
// note: also see xfunctors.h - functors implemented there can
// use Eigen's internal support for vectorized operations which
// can give faster performance on some hardware

namespace dynet {

struct FHuberForward {
  FHuberForward(float c) : c(c) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    const float a = fabs(x);
    return (a < c) ? x*x : c*(2*a - c);
  }
  const float c;
};

// template <typename T> int sgn(T val) {
//   return ((T(0) < val) - (val < T(0)));
// }

struct FL1Backward {
  FL1Backward(float d) : d(d) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return ((0.f < x) - (x < 0.f)) * d;
  }
  const float d;
};

struct FHuberBackward {
  FHuberBackward(float c, float dEdf) : c(c), d(dEdf) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    const float a = fabs(x);
    return (2 * d) * ((a < c) ? x : c * ((0.f < x) - (x < 0.f)));
  }
  const float c;
  const float d;
};

struct FProduct {
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    return a * b;
  }
};

struct FQuotient {
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    return a / b;
  }
};

struct FConstantPlus {
  FConstantPlus(float c) : c(c) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return c + x;
  }
  float c;
};

struct FConstantMinus {
  FConstantMinus(float c) : c(c) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return c - x;
  }
  float c;
};

struct FNegate {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return -x;
  }
};

struct FErf {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return erff(x);
  }
};

struct FTanh {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
#ifdef FAST_TANH
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
#else
     return tanhf(x);
#endif
  }
};

struct FLog {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return logf(x);
  }
};

struct FMaxBackwardInv {
  DYNET_DEVICE_FUNC inline float operator()(float u, float d) const {
    return (1.f - u) * d;
  }
};

struct FSqrtBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return d / (2.f * t);
  }
};

struct FErfBackward {
  DYNET_DEVICE_FUNC inline float operator()(float x, float d) const {
    return 1.1283791670955125738961589f * expf(-x * x) * d;
  }
};

struct FTanhBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f - t * t) * d;
  }
};

struct FLogBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f / t) * d;
  }
};

struct FPairwiseRankLoss {
  FPairwiseRankLoss(float m) : margin(m) {}
  DYNET_DEVICE_FUNC float operator()(float a, float b) const {
    float d = margin - a + b;
    return d > 0.f ? d : 0.f;
  }
  float margin;
};

struct FRectifyBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (t) ? d : 0.f;
  }
};

struct FRectifyNegateBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (t) ? -d : 0.f;
  }
};

struct FSoftmaxNormalize {
  explicit FSoftmaxNormalize(float logz) : logz(logz) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return expf(x - logz);
  }
  float logz;
};

struct FSoftmaxBackward {
  explicit FSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

struct FNegLogSoftmaxBackward {
  FNegLogSoftmaxBackward(float lz, float err) : logz(lz), d(err) {}
  DYNET_DEVICE_FUNC inline float operator()(float t) const {
    return expf(t - logz) * d;
  }
  float logz;
  float d;
};

struct FPtrNegLogSoftmaxBackward {
  FPtrNegLogSoftmaxBackward(const float* lz, const float* err) : logz(lz), d(err) {}
  DYNET_DEVICE_FUNC inline float operator()(float t) const {
    return expf(t - *logz) * *d;
  }
  const float* logz;
  const float* d;
};

struct FLogSoftmaxNormalize {
  explicit FLogSoftmaxNormalize(float logz) : logz(logz) {}
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return x - logz;
  }
  float logz;
};

struct FWeightedError {
  float operator()(float t, float d) const {
    return expf(t) * d / expf(t);
  }
};

struct FLogSoftmaxBackward {
  explicit FLogSoftmaxBackward(float off_diag_sum) : off_diag_sum(off_diag_sum) {}
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return off_diag_sum * expf(t) + d;
    //return (off_diag_sum + d) * t;
  }
  float off_diag_sum;
};

struct FRectify {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return (x > 0.f) ? x : 0.f;
  }
};

struct FSoftSign {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return x / (1.f + (x < 0.f ? -x : x));
  }
};

struct FSoftSignBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    float a = 1.f - (t < 0.f ? -t : t);
    return a * a * d;
  }
};

struct FLogisticSigmoid {
  DYNET_DEVICE_FUNC inline float operator()(float x) const {
    return 1.f / (1.f + expf(-x));
  }
};

struct FLogisticSigmoidBackward {
  DYNET_DEVICE_FUNC inline float operator()(float t, float d) const {
    return (1.f - t) * t * d;
  }
};

struct FSqDist {
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    float d = a - b;
    return d * d;
  }
};

struct FEuclideanBackward {
  FEuclideanBackward(int i, const float* s) : i(i), scalar(s) {}
  DYNET_DEVICE_FUNC inline float operator()(float a, float b) const {
    return (i == 0 ? 2.f : -2.f) * (*scalar) * (a - b);
  }
  int i;
  const float* scalar;
};

struct FL2SGDUpdate {
  FL2SGDUpdate(float l, float s) : lambda(l), scale(-s) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float g) const {
    return scale * g - x * lambda;
  }
  float lambda;
  float scale;
};

struct FBinaryLogLoss {
  DYNET_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x_true == 1.f) {
      if (x == 0.f) return -1.f * log(DYNET_DEVICE_MIN);
      return -1.f * log(x);
    }
    else if (x_true == 0.f) {
      if (x == 1.f) return -1.f * log(DYNET_DEVICE_MIN);
      else return (x_true - 1.f) * log1p(-x);
    }
    else {
      if (x == 0.f) return -1.f * log(DYNET_DEVICE_MIN);
      else if (x == 1.f) return -1.f * log(DYNET_DEVICE_MIN);
      else return -1.f * (x_true * log(x) + (1.f - x_true) * log1p(-x));
    }
  }
};

struct FBinaryLogLossBackward {
  explicit FBinaryLogLossBackward(float d) : d(d) {}
  DYNET_DEVICE_FUNC inline float operator()(float x, float x_true) const {
    if (x == x_true) return 0;
    if (x == 0.f) x = DYNET_DEVICE_MIN;
    if (x == 1.f) x = 0.9999999f;
    if (x_true == 1.f) {
      return d * -x_true / x;
    } else if (x_true == 0.f) {
      return d * (1.f - x_true) / (1.f - x);
    }
    return d * ((1.f - x_true) / (1.f - x) + (-x_true / x));
  }
  float d;
};

} // namespace dynet

#endif
