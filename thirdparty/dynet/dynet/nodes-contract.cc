#include "dynet/nodes-contract.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#include "dynet/nodes-macros.h"

// This file takes a long time to compile on GPU. Uncomment this line to skip it.
#define DYNET_SKIP_CUDA_CONTRACTIONS

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dot(" << arg_names[0] << "," << arg_names[1] << ')';
  if (arg_names.size() == 3) s << " + " << arg_names[2];
  return s.str();
}

Dim InnerProduct3D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3)
    throw std::invalid_argument("Expected two or three arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      !LooksLikeVector(xs[1]) ||
      xs[0].size(2) != xs[1].size(0)) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0), xs[0].size(1)}, max(xs[0].bd, xs[1].bd));
  if(xs.size() == 3) d.bd = max(d.bd, xs[2].bd);
  if (xs.size() == 3 && xs[2] != d) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

string InnerProduct3D_1D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dotdot(" << arg_names[0] << "," << arg_names[1] << "," << arg_names[2] << ')';
  if (arg_names.size() == 4) s << " + " << arg_names[3];
  return s.str();
}

Dim InnerProduct3D_1D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 3 && xs.size() != 4)
    throw std::invalid_argument("Expected three or four arguments in InnerProduct3D_1D");
  if (xs[0].ndims() != 3 ||
      !LooksLikeVector(xs[1]) ||
      !LooksLikeVector(xs[2])) {
    // TODO fix add check
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  Dim d({xs[0].size(0)}, max(max(xs[0].bd, xs[1].bd), xs[2].bd));
  if(xs.size() == 4) d.bd = max(d.bd, xs[3].bd);
  if (xs.size() == 4 && xs[3] != d) {
    ostringstream s; s << "Bad input dimensions in InnerProduct3D_1D_1D: " << xs;
    throw std::invalid_argument(s.str());
  }
  return d;
}

#endif

//   Y_ij = A_ijk * B_k (+ C_ij)
template<class MyDevice>
void InnerProduct3D_1D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D::forward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto A = xs[0]->t<3>();
  auto b = xs[1]->t<1>();
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  if (xs.size() == 2) {
    fx.t<2>().device(*dev.edevice) = A.contract(b, dims);
  } else {
    auto C = xs[2]->t<2>();
    fx.t<2>().device(*dev.edevice) = A.contract(b, dims) + C;
  }
#endif
}

template<class MyDevice>
void InnerProduct3D_1D::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D::backward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto tdEdf = dEdf.t<2>();  // 2 tensor
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) { // 3 tensor
    // tensor product
    auto b = xs[1]->t<1>();
    dEdxi.t<3>().device(*dev.edevice) += tdEdf.contract(b, Eigen::array<DimPair, 0>{{}});
  } else if (i == 1) {
    auto A = xs[0]->t<3>();  // A is 3 tensor
    Eigen::array<DimPair, 2> dims({{DimPair(0, 0), DimPair(1, 1)}});
    dEdxi.t<1>().device(*dev.edevice) += tdEdf.contract(A, dims);
  } else if (i == 2) {
    dEdxi.t<2>().device(*dev.edevice) += tdEdf;
  } else {
    throw std::runtime_error("Illegal configuration in InnerProduct3D");
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(InnerProduct3D_1D)

//   Y_ij = A_ijk * B_k * C_j (+ D_i)
template<class MyDevice>
void InnerProduct3D_1D_1D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D_1D::forward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto A = xs[0]->t<3>();
  auto b = xs[1]->t<1>();
  auto c = xs[2]->t<1>();
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
  Eigen::array<DimPair, 1> dims2({{DimPair(1, 0)}});
  if (xs.size() == 3) {
    fx.t<1>().device(*dev.edevice) = A.contract(b, dims).contract(c, dims2);
  } else {
    auto d = xs[3]->t<1>();
    fx.t<1>().device(*dev.edevice) = A.contract(b, dims).contract(c, dims2) + d;
  }
#endif
}

template<class MyDevice>
void InnerProduct3D_1D_1D::backward_dev_impl(const MyDevice & dev,
                                             const vector<const Tensor*>& xs,
                                             const Tensor& fx,
                                             const Tensor& dEdf,
                                             unsigned i,
                                             Tensor& dEdxi) const {
#if defined(__CUDACC__) && defined(DYNET_SKIP_CUDA_CONTRACTIONS)
  throw std::runtime_error("InnerProduct3D_1D_1D::backward_dev_impl disabled on CUDA. Comment out DYNET_SKIP_CUDA_CONTRACTIONS in nodes-contract.cc to enable this function.");
#else
  auto tdEdf = dEdf.t<1>();  // vector
  typedef Eigen::Tensor<float, 1>::DimensionPair DimPair;
  if (i == 0) { // 3 tensor
    // tensor product
    auto b = xs[1]->t<1>();
    auto c = xs[2]->t<1>();
    dEdxi.t<3>().device(*dev.edevice) += tdEdf.contract(c, Eigen::array<DimPair, 0>{{}}).contract(b, Eigen::array<DimPair, 0>{{}});
  } else if (i == 1) { // vector 1
    // TODO these should be reorganized so the contraction is first with tdEdf and then with c or b.
    // in theory, that intermediate result could be cached (although DYNET doesn't support this). the fact that it
    // this part of the product is redone when i=1 and again when i=2 is probably why this is slower
    // (or maybe it's the contract implementation?)
    Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto c = xs[2]->t<1>();
    dEdxi.t<1>().device(*dev.edevice) += A.contract(c, dims).contract(tdEdf, dims2);
  } else if (i == 2) { // vector 2
    Eigen::array<DimPair, 1> dims({{DimPair(2, 0)}});
    Eigen::array<DimPair, 1> dims2({{DimPair(0, 0)}});
    auto A = xs[0]->t<3>();
    auto b = xs[1]->t<1>();
    dEdxi.t<1>().device(*dev.edevice) += A.contract(b, dims).contract(tdEdf, dims2);
  } else if (i == 3) { // vector bias
    dEdxi.t<1>().device(*dev.edevice) += tdEdf;
  } else {
    throw std::runtime_error("Illegal configuration in InnerProduct3D");
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(InnerProduct3D_1D_1D)


} // namespace dynet
