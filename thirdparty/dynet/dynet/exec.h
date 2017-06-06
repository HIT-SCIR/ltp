#ifndef DYNET_EXEC_H
#define DYNET_EXEC_H

#include "dynet/dynet.h"

namespace dynet {

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine();
  virtual void invalidate() = 0;
  virtual void invalidate(unsigned) = 0;
  virtual const Tensor& forward() = 0;
  virtual const Tensor& forward(VariableIndex i) = 0;
  virtual const Tensor& incremental_forward() = 0;  // if you want to add nodes and evaluate just the new parts
  virtual const Tensor& incremental_forward(VariableIndex i) = 0;
  virtual const Tensor& get_value(VariableIndex i) = 0;
  virtual const Tensor& get_gradient(VariableIndex i) = 0;
  virtual void backward(bool full = false) = 0;
  virtual void backward(VariableIndex i, bool full = false) = 0;
 protected:
  explicit ExecutionEngine(const ComputationGraph& cg) : cg(cg) {}
  const ComputationGraph& cg;
  VariableIndex backward_computed;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(const ComputationGraph& cg) : ExecutionEngine(cg) {}
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;  // if you want to add nodes and evaluate just the new parts
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  void backward(bool full = false) override;
  void backward(VariableIndex i, bool full = false) override;
 private:
  std::vector<Tensor> nfxs;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated;
};

} // namespace dynet

#endif
