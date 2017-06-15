#if HAVE_CUDNN
#include <iostream>
#include <vector>
#include <algorithm>

#include "dynet/dynet.h"
#include "dynet/cudnn-ops.h"

namespace dynet {

CudnnConvOp::CudnnConvOp(const std::vector<unsigned>& s, const bool padding_type) {
  stride.resize(s.size());
  for (unsigned i = 0; i < stride.size(); ++i) {
    stride[i] = static_cast<int>(s[i]);
  }
  is_valid = padding_type;

  fwd_workspace = NULL;
  bwd_filter_workspace = NULL;
  bwd_data_workspace = NULL;
  workspace_fwd_size_ = 0;
  workspace_bwd_data_size_ = 0;
  workspace_bwd_filter_size_ = 0;
  mempool_ = NULL;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
}

CudnnConvOp::~CudnnConvOp() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
}

void CudnnConvOp::forward_impl(const Device_GPU & dev, const std::vector<const Tensor*>& xs, Tensor& fx) {
  const Tensor* x = xs[0]; 
  const Tensor* filter = xs[1];
  Tensor* y = &fx;
 
  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  unsigned FYC = filter->d[3];
  unsigned FXC = filter->d[2];
  unsigned FH = filter->d[0];
  unsigned FW = filter->d[1];
  unsigned YN = fx.d.bd;
  unsigned YC = fx.d[2];
  unsigned YH = fx.d[0];
  unsigned YW = fx.d[1];

  // infer pad_h, pad_w
  if (!is_valid) {
  // Total padding on rows and cols is
  // Pr = (R' - 1) * S + Kr - R
  // Pc = (C' - 1) * S + Kc - C
  // where (R', C') are output dimensions, (R, C) are input dimensions, S
  // is stride, (Kr, Kc) are filter dimensions.
  // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
  // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
  // we pad more on the right and bottom than on the top and left.
    pad_h = std::max<int>(0, (YH - 1) * stride[0] + FH - XH);
    pad_w = std::max<int>(0, (YW - 1) * stride[1] + FW - XW);
    if (mempool_ == NULL) {
      throw std::runtime_error("dynet::CudnnConvOp::mempool_ not set");
    }
    const bool h_odd = (pad_h % 2 != 0);
    const bool w_odd = (pad_w % 2 != 0);
    if (h_odd || w_odd) { // then we need to pad one row/col on the bottom/right
      unsigned new_XH = XH + h_odd;
      unsigned new_XW = XW + w_odd;
      void* temp = mempool_->allocate(sizeof(float) * new_XW * new_XH * XC * XN);
      padded_x = Tensor(Dim({ new_XH, new_XW, XC }, XN), static_cast<float*>(temp), xs[0]->device, DeviceMempool::FXS);
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = std::make_pair(0, static_cast<int>(h_odd));
      paddings[1] = std::make_pair(0, static_cast<int>(w_odd));
      paddings[2] = std::make_pair(0, 0);
      paddings[3] = std::make_pair(0, 0);
      padded_x.tb<3>().device(*dev.edevice) = xs[0]->tb<3>().pad(paddings);
      XH = new_XH;
      XW = new_XW;
      x = &padded_x;
    }
  }

  if (xs.size() == 3) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, 
                CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
                1, FYC, 1, 1));
  }
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc_, 
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              XN, XC, XW, XH));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc_, 
              CUDNN_TENSOR_NCHW, DataTypeToCudnnType<float>::value,
              YN, YC, YW, YH));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, 
              DataTypeToCudnnType<float>::value, CUDNN_TENSOR_NCHW,
              FYC, FXC, FW, FH));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
              pad_w/2, pad_h/2, stride[1], stride[0], 1, 1, 
              CUDNN_CROSS_CORRELATION));

  //TODO(Hao Zhang): there should be an autotune function to determine
  // the best convolution algorithm to use.
  // However, as DyNet changes CG for every sample (or every iteration),
  // This autotune function seems to be unnecessary.
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(dev.cudnnHandle,
              x_desc_, filter_desc_, conv_desc_, y_desc_,
              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, workspace_size_limit_bytes,
              &fwd_algo_));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(dev.cudnnHandle,
              x_desc_, filter_desc_, conv_desc_, y_desc_,
              fwd_algo_, &workspace_fwd_size_));
  if (fwd_workspace == NULL) {
    fwd_workspace = mempool_->allocate(workspace_fwd_size_);
  }
  float alpha = 1.f, beta = 0.f;
  CUDNN_CHECK(cudnnConvolutionForward(dev.cudnnHandle,
              &alpha, x_desc_, x->v, filter_desc_, filter->v,
              conv_desc_, fwd_algo_, fwd_workspace, workspace_fwd_size_,
              &beta, y_desc_, y->v));
  if (xs.size() == 3) {
    CUDNN_CHECK(cudnnAddTensor(dev.cudnnHandle, &alpha, 
                bias_desc_, xs[2]->v, &alpha, y_desc_, y->v));
  }
}

void CudnnConvOp::backward_impl(const Device_GPU & dev, 
             const std::vector<const Tensor*>& xs,
             const Tensor& fx,
             const Tensor& dEdf,
             unsigned i,
             Tensor& dEdxi) {
  const Tensor* x = xs[0]; 
  const Tensor* filter = xs[1];
  const Tensor* dy = &dEdf;
  Tensor* dxi = &dEdxi;
  unsigned XN = x->d.bd;
  unsigned XC = x->d[2];
  unsigned XH = x->d[0];
  unsigned XW = x->d[1];
  const bool h_odd = (pad_h % 2 != 0);
  const bool w_odd = (pad_w % 2 != 0);
  void* dx_ptr = NULL;
  if (mempool_ == NULL) 
    throw std::runtime_error("dynet::CudnnConvOp::mempool_ not set");
  if (h_odd || w_odd) {
    unsigned new_XH = XH + h_odd;
    unsigned new_XW = XW + w_odd;
    DYNET_ASSERT(padded_x.d[0] == new_XH, "Tensor input_padded must have been padded");
    DYNET_ASSERT(padded_x.d[1] == new_XW, "Tensor input_padded must have been padded");
    x = &padded_x;
    XH = new_XH;
    XW = new_XW;
    if (i == 0)
      dx_ptr = mempool_->allocate(sizeof(float) * new_XW * new_XH * XC * XN);
  }
  // here we could reuse the descriptor we created for forward, because 
  // they share the same size
  float alpha = 1.f, beta = 0.f;
  if (i == 1) {
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(dev.cudnnHandle,
                x_desc_, y_desc_, conv_desc_, filter_desc_, 
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 
                workspace_size_limit_bytes, &bwd_f_algo_));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(dev.cudnnHandle,
                x_desc_, y_desc_, conv_desc_, filter_desc_,
                bwd_f_algo_, &workspace_bwd_filter_size_));
    // allocate space for backward compute
    if (bwd_filter_workspace == NULL) {
      bwd_filter_workspace = mempool_->allocate(sizeof(float) * workspace_bwd_filter_size_);
    }
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(dev.cudnnHandle,
                &alpha, x_desc_, x->v,
                y_desc_, dy->v,
                conv_desc_, bwd_f_algo_, bwd_filter_workspace, workspace_bwd_filter_size_,
                &beta, filter_desc_, dxi->v));
  } else if (i == 0) {            
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(dev.cudnnHandle,
                filter_desc_, y_desc_, conv_desc_, x_desc_,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                workspace_size_limit_bytes, &bwd_d_algo_));
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(dev.cudnnHandle,
                filter_desc_, y_desc_, conv_desc_, x_desc_,
                bwd_d_algo_, &workspace_bwd_data_size_));
    if (bwd_data_workspace == NULL) {
      bwd_data_workspace = mempool_->allocate(sizeof(float) * workspace_bwd_data_size_);
      }
    if (h_odd || w_odd) {
      CUDNN_CHECK(cudnnConvolutionBackwardData(dev.cudnnHandle,
                  &alpha, filter_desc_, filter->v,
                  y_desc_, dy->v,
                  conv_desc_, bwd_d_algo_, bwd_data_workspace, workspace_bwd_data_size_,
                  &beta, x_desc_, dx_ptr));
      Tensor padded_dx = Tensor(Dim({XH, XW, XC}, XN), static_cast<float*>(dx_ptr), xs[0]->device, DeviceMempool::FXS);
        
      Eigen::array<int, 4> offsets = {0, 0, 0, 0};
      Eigen::array<int, 4> extents = {static_cast<int>(XH), static_cast<int>(XW), static_cast<int>(XC), static_cast<int>(XN)};
      dxi->tb<3>().device(*dev.edevice) = padded_dx.tb<3>().slice(offsets, extents);
    } else {
      CUDNN_CHECK(cudnnConvolutionBackwardData(dev.cudnnHandle,
                  &alpha, filter_desc_, filter->v,
                  y_desc_, dy->v,
                  conv_desc_, bwd_d_algo_, bwd_data_workspace, workspace_bwd_data_size_,
                  &beta, x_desc_, dxi->v));
    }
  } else {
    CUDNN_CHECK(cudnnConvolutionBackwardBias(dev.cudnnHandle,
                &alpha, y_desc_, dy->v,
                &beta, bias_desc_, dxi->v));  
  }
}

} // namespace dynet

#endif
