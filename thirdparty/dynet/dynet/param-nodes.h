#ifndef DYNET_PARAM_NODES_H_
#define DYNET_PARAM_NODES_H_

#include "dynet/dynet.h"
#include "dynet/model.h"
#include "dynet/nodes-macros.h"

namespace dynet {

struct ParameterNodeBase : public Node {
  virtual void accumulate_grad(const Tensor& g) = 0;
};

// represents optimizable parameters
struct ParameterNode : public ParameterNodeBase {
  explicit ParameterNode(const Parameter & p) : dim(p.get()->dim), params(p) {}
  explicit ParameterNode(const LookupParameter & lp) : dim(lp.get()->all_dim), lparams(lp) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  Parameter params;
  LookupParameter lparams;
};

// represents optimizable parameters that are being held constant
struct ConstParameterNode : public Node {
  explicit ConstParameterNode(const Parameter & p) : dim(p.get()->dim), params(p) {}
  explicit ConstParameterNode(const LookupParameter & lp) : dim(lp.get()->all_dim), lparams(lp) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  Parameter params;
  LookupParameter lparams;
};

// represents specified (not learned) inputs to the network
struct InputNode : public Node {
  explicit InputNode(const Dim& d, const std::vector<float>& dat) : dim(d), data(dat), pdata(&data) {}
  explicit InputNode(const Dim& d, const std::vector<float>* pdat) : dim(d), data(), pdata(pdat) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  Dim dim;
  const std::vector<float> data;
  const std::vector<float>* pdata;
};

// Represents specified (not learned) inputs to the network in sparse array format,
// with an optional default value. Note that indexes refer to where the memory is actually
// indexed in column-major format. When multiple batches are used they will also be
// consecutive in memory. This doesn't support pointer input, because this would require
// dynamic changing of the size of auxiliary memory on GPUs, although this could possibly
// be fixed in the future.
struct SparseInputNode : public Node {
  explicit SparseInputNode(const Dim& d, const std::vector<unsigned int>& id, const std::vector<float>& dat, float defdat = 0.f) : dim(d), ids(id), data(dat), defdata(defdat) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  Dim dim;
  const std::vector<unsigned int> ids;
  const std::vector<float> data;
  float defdata;
};

// represents specified (not learned) scalar inputs to the network
struct ScalarInputNode : public Node {
  explicit ScalarInputNode(real s) : data(s), pdata(&data) {}
  explicit ScalarInputNode(const real* ps) : data(), pdata(ps) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  const dynet::real data;
  const dynet::real* pdata;
};

// represents a matrix/vector embedding of an item of a discrete set (1-hot coding)
struct LookupNode : public ParameterNodeBase {
  LookupNode(LookupParameter p, unsigned ind) : dim(p.get()->dim), index(ind), pindex(&index), indices(), pindices(), params(p) {}
  LookupNode(LookupParameter p, const unsigned* pind) : dim(p.get()->dim), index(), pindex(pind), indices(), pindices(), params(p) {}
  LookupNode(LookupParameter p, const std::vector<unsigned>& indices) : dim(p.get()->dim), index(), pindex(), indices(indices), pindices(&this->indices), params(p) { dim.bd = pindices->size(); }
  LookupNode(LookupParameter p, const std::vector<unsigned>* pindices) : dim(p.get()->dim), index(), pindex(), indices(), pindices(pindices), params(p) { dim.bd = pindices->size(); }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }  
  size_t aux_storage_size() const override;
  void accumulate_grad(const Tensor& g) override;
  Dim dim;
  unsigned index;
  const unsigned* pindex;
  std::vector<unsigned> indices;
  const std::vector<unsigned>* pindices;
  LookupParameter params;
};

} // namespace dynet

#endif
