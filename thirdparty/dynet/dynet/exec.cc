#include "dynet/exec.h"

#include "dynet/param-nodes.h"
#include "dynet/globals.h"

using namespace std;

namespace dynet {

ExecutionEngine::~ExecutionEngine() {}

void SimpleExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  backward_computed = 0;
}

void SimpleExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& SimpleExecutionEngine::forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& SimpleExecutionEngine::get_value(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::get_value()");
  if (i >= num_nodes_evaluated) {
    incremental_forward();
  }
  return nfxs[i];
}

const Tensor& SimpleExecutionEngine::get_gradient(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::get_value()");
  if (i >= backward_computed) {
    DYNET_RUNTIME_ERR("Requested gradient for node " << i << ", but backward pass was computed from node " << (backward_computed - 1));
  }
  return ndEdfs[i];
}

const Tensor& SimpleExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(), "Out-of-bounds variable access in SimpleExecutionEngine::incremental_forward()");

  // free any old memory if this is a new CG
  if (num_nodes_evaluated == 0)
    for(Device* dev : dynet::devices)
      dev->pools[(int)DeviceMempool::FXS]->free();

  if (i >= num_nodes_evaluated) {
    nfxs.resize(i + 1);

    //vector<string> dummy(5, "x");
    vector<const Tensor*> xs(16);
    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
      const Node* node = cg.nodes[num_nodes_evaluated];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        xs[ai] = &nfxs[arg];
        ++ai;
      }
      nfxs[num_nodes_evaluated].d = node->dim;
      // Get the device
      DYNET_ASSERT(node->device != nullptr, "Attempt to access null device in SimpleExecutionEngine::incremental_forward");
      nfxs[num_nodes_evaluated].device = node->device;
      nfxs[num_nodes_evaluated].mem_pool = DeviceMempool::FXS;
      // Get the memory
      nfxs[num_nodes_evaluated].v = static_cast<float*>(nfxs[num_nodes_evaluated].device->pools[(int)DeviceMempool::FXS]->allocate(node->dim.size() * sizeof(float)));
      if (nfxs[num_nodes_evaluated].v == nullptr)
        DYNET_RUNTIME_ERR("Ran out of memory when executing node " << num_nodes_evaluated);
      void* aux_mem = nullptr;
      size_t aux_size = node->aux_storage_size();
      if (aux_size) {
        aux_mem = nfxs[num_nodes_evaluated].device->pools[(int)DeviceMempool::FXS]->allocate(aux_size);
        if (!aux_mem)
          DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node " << num_nodes_evaluated);
      }
      node->aux_mem = aux_mem;

      node->forward(xs, nfxs[num_nodes_evaluated]);
    }
  }
  return nfxs[i];
}

void SimpleExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(), "Mismatched array sizes in SimpleExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size()-1),full);
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void SimpleExecutionEngine::backward(VariableIndex from_where, bool full) {
  if(!(from_where < nfxs.size()))
    incremental_forward(from_where);
  if (nfxs[from_where].d.size() != 1)
    DYNET_INVALID_ARG("backward() can only be called on scalar nodes, but node " << from_where << " has dimension: " << nfxs[from_where].d);

  const unsigned num_nodes = from_where+1;
  ndEdfs.resize(num_nodes);
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].device = nfxs[i].device;
    ndEdfs[i].mem_pool = DeviceMempool::DEDFS;
    ndEdfs[i].v = static_cast<float*>(ndEdfs[i].device->pools[(int)DeviceMempool::DEDFS]->allocate(dim.size() * sizeof(float)));
    if (!ndEdfs[i].v)
      DYNET_RUNTIME_ERR("out of memory while attempting to allocate space for derivatives of node " << i);
  }
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();
  // initialize dE/dE = 1
  ndEdfs.back().v = kSCALAR_ONE;

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      needs_derivative[i] = true;

    for (unsigned ni = 0; ni < num_nodes; ++ni) {
      bool nd = needs_derivative[ni];
      for (auto arg : cg.nodes[ni]->args)
        nd |= needs_derivative[arg];
      needs_derivative[ni] = nd;
    }
  }

  // loop in reverse topological order
  // consider only nodes that participate in the computation.
  vector<bool> in_computation(num_nodes, false);
  in_computation[num_nodes - 1] = true;
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      in_computation[arg] = true;
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
      if (needs_derivative[arg]) {
        node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
  // We use this because from_where + 1 because 0 corresponds to "backward wasn't computed"
  backward_computed =  from_where + 1;
}

} // namespace dynet
