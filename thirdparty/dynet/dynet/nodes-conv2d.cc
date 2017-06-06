#include "dynet/nodes-conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <array>

#include "dynet/functors.h"
#include "dynet/nodes-macros.h"
#include "third_party/eigen_spatial_convolutions.h"
#include "third_party/eigen_backward_spatial_convolutions.h"

#if HAVE_CUDA
#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#endif

using namespace std;

namespace dynet {

#ifndef __CUDACC__

string Conv2D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "conv2d(" << arg_names[0] << ", f=" << arg_names[1];
  if (arg_names.size() == 3)
    s << ", b=" << arg_names[2];
  s << ")";
  return s.str();
}

Dim Conv2D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3) {
    ostringstream s; s << "Conv2D requires either two or three inputs: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs[0].ndims() != 3 || xs[1].ndims() != 4 ||
      xs[1].d[2] != xs[0].d[2]) {
    ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
    throw std::invalid_argument(s.str());
  }
  if (is_valid && (xs[0].d[0] < xs[1].d[0] || xs[0].d[1] < xs[1].d[1])) {
    ostringstream s; s << "Bad input dimensions in Conv2D: in VALID convolution, the filter size must not be greater than the feature map size" << xs;
    throw std::invalid_argument(s.str());
  }
  if (xs.size() == 3) { //has bias term
    if (xs[2].d[0] != xs[1].d[3] || xs[2].ndims() != 1) {
      ostringstream s; s << "Bad input dimensions in Conv2D: " << xs;
      throw std::invalid_argument(s.str());
    }
  }
  unsigned bs = xs[0].batch_elems();
  std::vector<long> output_shape(3);
  output_shape[2] = static_cast<long>(xs[1].d[3]);
  for (unsigned i = 0; i < 2; ++i) {
    float input_dim = static_cast<float>(xs[0].d[i]);
    float kernel_dim = static_cast<float>(xs[1].d[i]);
    float s = static_cast<float>(stride[i]);
    if (is_valid) {
      output_shape[i] = static_cast<long>(ceil((input_dim - kernel_dim + 1) / s));
    } else {
      output_shape[i] = static_cast<long>(ceil(input_dim / s));
    }
  }
  return Dim(output_shape, bs);
}

size_t Conv2D::aux_storage_size() const {
  vector<unsigned> input_size(arity());
  for (unsigned i = 0; i < arity(); ++i) {
    input_size[i] = get_cg()->nodes[args[i]]->dim.size();
  }
  size_t nbytes = 0;
#if HAVE_CUDNN
  nbytes += CudnnConvOp::workspace_size_limit_bytes;
  nbytes += 3 * input_size[0] * sizeof(float);
#else
  nbytes += sizeof(float) * (input_size[0] + input_size[1] + 
      dim.size() + std::max(input_size[0], input_size[1]));
#endif
  return nbytes;
}
#endif

template<class MyDevice>
void Conv2D::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2 || xs.size() == 3, "Failed dimension check in Conv2D::forward, at least 2 inputs");
  DYNET_ASSERT(fx.d.bd == xs[0]->d.bd, "Failed dimension check in Conv2D::forward, batchsize not match");
  DYNET_ASSERT(fx.d[2] == xs[1]->d[3], "Failed dimension check in Conv2D::forward, #channel not match");
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
#ifdef __CUDACC__
#if HAVE_CUDNN
  if (cudnn_conv_op_ == NULL) {
    cudnn_conv_op_ = new CudnnConvOp(stride, is_valid);
    cudnn_conv_op_->set_pool(&aux_mem_pool);
  }
  cudnn_conv_op_->forward_impl(dev, xs, fx);
#else
  throw std::runtime_error("Conv2D::forward_dev_impl not supported without CUDNN");
#endif
#else
  Eigen::PaddingType padding_type = is_valid ? Eigen::PADDING_VALID : Eigen::PADDING_SAME;
  void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
  Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles; 
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  CHWN_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().shuffle(shuffles);
  void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
  Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
  shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
  NCHW_f.t<4>().device(*dev.edevice) = xs[1]->t<4>().shuffle(shuffles);
  void* CHWN_y_mem = aux_mem_pool.allocate(fx.d.size() * sizeof(float));
  Tensor CHWN_y = Tensor(Dim({fx.d[2], fx.d[0], fx.d[1]}, fx.d.bd), static_cast<float*>(CHWN_y_mem), fx.device, DeviceMempool::FXS);
  CHWN_y.tb<3>().device(*dev.edevice) = Eigen::SpatialConvolution(CHWN_x.tb<3>(), NCHW_f.t<4>(), stride[0], stride[1], padding_type);
  shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
  fx.tb<3>().device(*dev.edevice) = CHWN_y.tb<3>().shuffle(shuffles);
  //NWHCToNCWH()(&NWHC_y, fx);
  if (xs.size() == 3) {
    Tensor bias = Tensor(Dim({fx.d[0], fx.d[1], fx.d.bd}, 1), static_cast<float*>(CHWN_x_mem), xs[2]->device, DeviceMempool::FXS);
    for (unsigned i = 0; i < fx.d[2]; ++i) {
      TensorTools::constant(bias, xs[2]->vec()(i));
      fx.tb<3>().chip<2>(i).device(*dev.edevice) += bias.t<3>(); 
    }
  }
#endif
}

template<class MyDevice>
void Conv2D::backward_dev_impl(const MyDevice & dev,
                         const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
  // don't check those already checked in forward_impl
  DYNET_ASSERT(dEdf.d == fx.d, "Failed dimension check in Conv2D::backward");
  DYNET_ASSERT(dEdxi.d == xs[i]->d, "Failed dimension check in Conv2D::backward");
  DYNET_ASSERT(i <= 2, "Failed dimension check in Conv2D::backward");
  NodeMemPool aux_mem_pool = NodeMemPool(aux_storage_size(), aux_mem);
#ifdef __CUDACC__
#if HAVE_CUDNN
  DYNET_ASSERT(cudnn_conv_op_ != NULL, "cudnn operator is not initialized");
  cudnn_conv_op_->set_pool(&aux_mem_pool);
  cudnn_conv_op_->backward_impl(dev, xs, fx, dEdf, i, dEdxi);
#else
  throw std::runtime_error("Conv2D::backward_dev_impl not supported without CUDNN");
#endif
#else
  void* CHWN_dy_mem = aux_mem_pool.allocate(dEdf.d.size() * sizeof(float));
  Tensor CHWN_dy = Tensor(Dim({dEdf.d[2], dEdf.d[0], dEdf.d[1]}, dEdf.d.bd), static_cast<float*>(CHWN_dy_mem), dEdf.device, DeviceMempool::FXS);
  Eigen::array<ptrdiff_t, 4> shuffles; 
  shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
  CHWN_dy.tb<3>().device(*dev.edevice) = dEdf.tb<3>().shuffle(shuffles);
  if (i == 0) { //backward w.r.t the input
    void* NCHW_f_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_f = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_f_mem), xs[1]->device, DeviceMempool::FXS);
    shuffles[0] = 3; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 1;
    NCHW_f.t<4>().device(*dev.edevice) = xs[1]->t<4>().shuffle(shuffles);
    void* CHWN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_dEdxi = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    CHWN_dEdxi.tb<3>().device(*dev.edevice) = Eigen::SpatialConvolutionBackwardInput(NCHW_f.t<4>(), CHWN_dy.tb<3>(), xs[0]->d[0], xs[0]->d[1], stride[0], stride[1]);
    void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[0]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 1; shuffles[1] = 2; shuffles[2] = 0; shuffles[3] = 3;
    HWCN_dEdxi.tb<3>().device(*dev.edevice) = CHWN_dEdxi.tb<3>().shuffle(shuffles);
    dEdxi.tb<3>().device(*dev.edevice) += HWCN_dEdxi.tb<3>();
  } else if (i == 1) { //backward w.r.t the kernel
    void* CHWN_x_mem = aux_mem_pool.allocate(xs[0]->d.size() * sizeof(float));
    Tensor CHWN_x = Tensor(Dim({xs[0]->d[2], xs[0]->d[0], xs[0]->d[1]}, xs[0]->d.bd), static_cast<float*>(CHWN_x_mem), xs[0]->device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
    CHWN_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().shuffle(shuffles);
    void* NCHW_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor NCHW_dEdxi = Tensor(Dim({xs[1]->d[3], xs[1]->d[2], xs[1]->d[0], xs[1]->d[1]}), static_cast<float*>(NCHW_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    NCHW_dEdxi.t<4>().device(*dev.edevice) = Eigen::SpatialConvolutionBackwardKernel(CHWN_x.tb<3>(), CHWN_dy.tb<3>(), xs[1]->d[0], xs[1]->d[1], stride[0], stride[1]);
    void* HWCN_dEdxi_mem = aux_mem_pool.allocate(xs[1]->d.size() * sizeof(float));
    Tensor HWCN_dEdxi = Tensor(xs[1]->d, static_cast<float*>(HWCN_dEdxi_mem), dEdxi.device, DeviceMempool::FXS);
    shuffles[0] = 2; shuffles[1] = 3; shuffles[2] = 1; shuffles[3] = 0;
    HWCN_dEdxi.t<4>().device(*dev.edevice) = NCHW_dEdxi.t<4>().shuffle(shuffles);
    dEdxi.t<4>().device(*dev.edevice) += HWCN_dEdxi.t<4>();
  } else { //backward w.r.t the bias
    Eigen::array<int, 3> red_axis = {0, 1, 3};
    dEdxi.t<1>().device(*dev.edevice) += dEdf.tb<3>().sum(red_axis);
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(Conv2D)

} // namespace dynet 
