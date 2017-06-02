// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_builtins_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

namespace std {
template <typename T> T rsqrt(T x) { return 1 / std::sqrt(x); }
template <typename T> T square(T x) { return x * x; }
template <typename T> T cube(T x) { return x * x * x; }
template <typename T> T inverse(T x) { return 1 / x; }
}

#define TEST_UNARY_BUILTINS_FOR_SCALAR(FUNC, SCALAR, OPERATOR)                 \
  {                                                                            \
    /* out OPERATOR in.FUNC() */                                               \
    Tensor<SCALAR, 3> in(tensorRange);                                         \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    in = in.random() + static_cast<SCALAR>(0.01);                              \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3> reference(out);                                          \
    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
        sycl_device.allocate(in.size() * sizeof(SCALAR)));                     \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu(gpu_data, tensorRange);                   \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
                                   (in.size()) * sizeof(SCALAR));              \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) OPERATOR gpu.FUNC();                           \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < out.size(); ++i) {                                     \
      SCALAR ver = reference(i);                                               \
      ver OPERATOR std::FUNC(in(i));                                           \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data);                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }                                                                            \
  {                                                                            \
    /* out OPERATOR out.FUNC() */                                              \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3> reference(out);                                          \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) OPERATOR gpu_out.FUNC();                       \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < out.size(); ++i) {                                     \
      SCALAR ver = reference(i);                                               \
      ver OPERATOR std::FUNC(reference(i));                                    \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_UNARY_BUILTINS_OPERATOR(SCALAR, OPERATOR)                         \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR, OPERATOR)                        \
  TEST_UNARY_BUILTINS_FOR_SCALAR(sqrt, SCALAR, OPERATOR)                       \
  TEST_UNARY_BUILTINS_FOR_SCALAR(rsqrt, SCALAR, OPERATOR)                      \
  TEST_UNARY_BUILTINS_FOR_SCALAR(square, SCALAR, OPERATOR)                     \
  TEST_UNARY_BUILTINS_FOR_SCALAR(cube, SCALAR, OPERATOR)                       \
  TEST_UNARY_BUILTINS_FOR_SCALAR(inverse, SCALAR, OPERATOR)                    \
  TEST_UNARY_BUILTINS_FOR_SCALAR(tanh, SCALAR, OPERATOR)                       \
  TEST_UNARY_BUILTINS_FOR_SCALAR(exp, SCALAR, OPERATOR)                        \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log, SCALAR, OPERATOR)                        \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR, OPERATOR)                        \
  TEST_UNARY_BUILTINS_FOR_SCALAR(ceil, SCALAR, OPERATOR)                       \
  TEST_UNARY_BUILTINS_FOR_SCALAR(floor, SCALAR, OPERATOR)                      \
  TEST_UNARY_BUILTINS_FOR_SCALAR(round, SCALAR, OPERATOR)                      \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log1p, SCALAR, OPERATOR)

#define TEST_IS_THAT_RETURNS_BOOL(SCALAR, FUNC)                                \
  {                                                                            \
    /* out = in.FUNC() */                                                      \
    Tensor<SCALAR, 3> in(tensorRange);                                         \
    Tensor<bool, 3> out(tensorRange);                                          \
    in = in.random() + static_cast<SCALAR>(0.01);                              \
    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
        sycl_device.allocate(in.size() * sizeof(SCALAR)));                     \
    bool *gpu_data_out =                                                       \
        static_cast<bool *>(sycl_device.allocate(out.size() * sizeof(bool)));  \
    TensorMap<Tensor<SCALAR, 3>> gpu(gpu_data, tensorRange);                   \
    TensorMap<Tensor<bool, 3>> gpu_out(gpu_data_out, tensorRange);             \
    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
                                   (in.size()) * sizeof(SCALAR));              \
    gpu_out.device(sycl_device) = gpu.FUNC();                                  \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(bool));               \
    for (int i = 0; i < out.size(); ++i) {                                     \
      VERIFY_IS_EQUAL(out(i), std::FUNC(in(i)));                               \
    }                                                                          \
    sycl_device.deallocate(gpu_data);                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_UNARY_BUILTINS(SCALAR)                                            \
  TEST_UNARY_BUILTINS_OPERATOR(SCALAR, +=)                                     \
  TEST_UNARY_BUILTINS_OPERATOR(SCALAR, =)                                      \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isnan)                                     \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isfinite)                                  \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isinf)

static void test_builtin_unary_sycl(const Eigen::SyclDevice &sycl_device) {
  int sizeDim1 = 10;
  int sizeDim2 = 10;
  int sizeDim3 = 10;
  array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};

  TEST_UNARY_BUILTINS(float)
  /// your GPU must support double. Otherwise, disable the double test.
  TEST_UNARY_BUILTINS(double)
}

namespace std {
template <typename T> T cwiseMax(T x, T y) { return std::max(x, y); }
template <typename T> T cwiseMin(T x, T y) { return std::min(x, y); }
}

#define TEST_BINARY_BUILTINS_FUNC(SCALAR, FUNC)                                \
  {                                                                            \
    /* out = in_1.FUNC(in_2) */                                                \
    Tensor<SCALAR, 3> in_1(tensorRange);                                       \
    Tensor<SCALAR, 3> in_2(tensorRange);                                       \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    in_2 = in_2.random() + static_cast<SCALAR>(0.01);                          \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3> reference(out);                                          \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_2.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu_1(gpu_data_1, tensorRange);               \
    TensorMap<Tensor<SCALAR, 3>> gpu_2(gpu_data_2, tensorRange);               \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
                                   (in_2.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) = gpu_1.FUNC(gpu_2);                           \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < out.size(); ++i) {                                     \
      SCALAR ver = reference(i);                                               \
      ver = std::FUNC(in_1(i), in_2(i));                                       \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_2);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS_OPERATORS(SCALAR, OPERATOR)                       \
  {                                                                            \
    /* out = in_1 OPERATOR in_2 */                                             \
    Tensor<SCALAR, 3> in_1(tensorRange);                                       \
    Tensor<SCALAR, 3> in_2(tensorRange);                                       \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    in_2 = in_2.random() + static_cast<SCALAR>(0.01);                          \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3> reference(out);                                          \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_2.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu_1(gpu_data_1, tensorRange);               \
    TensorMap<Tensor<SCALAR, 3>> gpu_2(gpu_data_2, tensorRange);               \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
                                   (in_2.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) = gpu_1 OPERATOR gpu_2;                        \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < out.size(); ++i) {                                     \
      VERIFY_IS_APPROX(out(i), in_1(i) OPERATOR in_2(i));                      \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_2);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS_OPERATORS_THAT_TAKES_SCALAR(SCALAR, OPERATOR)     \
  {                                                                            \
    /* out = in_1 OPERATOR 2 */                                                \
    Tensor<SCALAR, 3> in_1(tensorRange);                                       \
    Tensor<SCALAR, 3> out(tensorRange);                                        \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    Tensor<SCALAR, 3> reference(out);                                          \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3>> gpu_1(gpu_data_1, tensorRange);               \
    TensorMap<Tensor<SCALAR, 3>> gpu_out(gpu_data_out, tensorRange);           \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) = gpu_1 OPERATOR 2;                            \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int i = 0; i < out.size(); ++i) {                                     \
      VERIFY_IS_APPROX(out(i), in_1(i) OPERATOR 2);                            \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS(SCALAR)                                           \
  TEST_BINARY_BUILTINS_FUNC(SCALAR, cwiseMax)                                  \
  TEST_BINARY_BUILTINS_FUNC(SCALAR, cwiseMin)                                  \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, +)                                    \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, -)                                    \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, *)                                    \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, /)

static void test_builtin_binary_sycl(const Eigen::SyclDevice &sycl_device) {
  int sizeDim1 = 10;
  int sizeDim2 = 10;
  int sizeDim3 = 10;
  array<int, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  TEST_BINARY_BUILTINS(float)
  TEST_BINARY_BUILTINS_OPERATORS_THAT_TAKES_SCALAR(int, %)
}

void test_cxx11_tensor_builtins_sycl() {
  cl::sycl::gpu_selector s;
  QueueInterface queueInterface(s);
  Eigen::SyclDevice sycl_device(&queueInterface);
  CALL_SUBTEST(test_builtin_unary_sycl(sycl_device));
  CALL_SUBTEST(test_builtin_binary_sycl(sycl_device));
}
