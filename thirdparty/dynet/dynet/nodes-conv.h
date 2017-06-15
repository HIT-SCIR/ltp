#ifndef DYNET_NODES_CONV_H_
#define DYNET_NODES_CONV_H_

#include "dynet/dynet.h"
#include "dynet/nodes-macros.h"
#include "dynet/op-helper.h"

#if HAVE_CUDNN
#include "dynet/cudnn-ops.h"
#endif

namespace dynet {

// with a single argument x \in R^{n x m}
// y_i = \sum_j x_i,j / m
struct AverageColumns : public Node {
  template <typename T> explicit AverageColumns(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

/* Deprecated
// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};
*/

// y = x_1 *filter x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Filter1DNarrow : public Node {
  explicit Filter1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned nrows;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1, unsigned dimension = 1) : Node(a), k(k), pooled_dim(dimension) {
    first_dim = pooled_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == pooled_dim ? first_dim + 2 : first_dim + 1;
  }
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned k;
  unsigned pooled_dim;
  unsigned first_dim;
  unsigned second_dim;
};

// sum along a single dimension
struct SumDimension : public Node {
  template <typename T> explicit SumDimension(const T& a, unsigned d) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned dimension;
};

// conv2d 
// y = x_1 *conv2d x_2
// x_1 \in R^{H x W x Ci x N} (input)
// x_2 \in R^{H x W x Ci x Co} (filter)
// stride[0] corresponds to H
// stride[1] corresponds to W
// is_valid: true for 'VALID' and false for 'SAME'
struct Conv2D: public Node {
  explicit Conv2D(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& s,
    const bool padding_type = true)
      : Node(a), stride(s), is_valid(padding_type) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  const std::vector<unsigned> stride;
  const bool is_valid;

 private:
#if HAVE_CUDNN
  mutable CudnnConvOp* cudnn_conv_op_ = NULL;
#endif
};


} // namespace dynet

#endif
