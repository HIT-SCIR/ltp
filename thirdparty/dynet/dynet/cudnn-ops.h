#ifndef DYNET_CUDNN_OPS_H
#define DYNET_CUDNN_OPS_H

#if HAVE_CUDNN
#include "dynet/dynet.h"
#include "dynet/cuda.h"
#include "dynet/op-helper.h"

namespace dynet {

class CudnnConvOp {
 public:
  explicit CudnnConvOp() {}
  explicit CudnnConvOp(const std::vector<unsigned>& s, const bool padding_type);
  ~CudnnConvOp();
  /* call this function before using the CudnnConvOp */
  void set_pool(NodeMemPool* mempool) {
    mempool_ = mempool;
  }
  void forward_impl(const Device_GPU & dev, const std::vector<const Tensor*>& xs, Tensor& fx);
  void backward_impl(const Device_GPU & dev, 
               const std::vector<const Tensor*>& xs,
               const Tensor& fx,
               const Tensor& dEdf,
               unsigned i,
               Tensor& dEdxi);
  static const size_t workspace_size_limit_bytes = 8 * 1024 * 1024;

 protected:
  std::vector<int> stride;
  bool is_valid;

  /* cuDNN resource */
  cudnnTensorDescriptor_t x_desc_, y_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_f_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_d_algo_;

  // cudnn workspace
  size_t workspace_fwd_size_;
  size_t workspace_bwd_data_size_;
  size_t workspace_bwd_filter_size_;
  void* fwd_workspace;
  void* bwd_filter_workspace;
  void* bwd_data_workspace;

 private:
  int pad_h = 0;
  int pad_w = 0;
  Tensor padded_x;
  Tensor padded_dx;
  NodeMemPool* mempool_;
};

/*
class CudnnMaxPoolingOp {

};
*/
} // namespace dynet

#endif
#endif
