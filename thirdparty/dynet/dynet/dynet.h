/**
 * \file dynet.h
 * \defgroup compgraph compgraph
 * \defgroup nodes nodes
 */
#ifndef DYNET_DYNET_H_
#define DYNET_DYNET_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <utility>

#include <boost/serialization/strong_typedef.hpp>

#include "dynet/init.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/tensor.h"
#include "dynet/model.h"
#include "dynet/devices.h"


namespace dynet {

extern float* kSCALAR_MINUSONE;
extern float* kSCALAR_ONE;
extern float* kSCALAR_ZERO;

/**
 * \ingroup compgraph
 * \brief Gets the number of active graphs
 * \details This is 0 or 1, you can't create more than one graph at once
 * \return Number of active graphs
 */
int get_number_of_active_graphs();
/**
 * \ingroup compgraph
 * \brief Get id of the current active graph
 * \details This can help check whether a graph is stale 
 * \return Id of the current graph
 */
unsigned get_current_graph_id();

// devices provide information about GPUs and CPUs
// these include any API information that is required to make calls
// to the GPU as well as the memory pools for the device
// Device is not copyable, so you can use the pointer to uniquely
// identify the device
//extern std::vector<Device*> devices; // [0] is always the CPU
extern Device* default_device; // where parameters go by default

class ExecutionEngine;
struct ParameterNodeBase;
struct Node;
namespace expr { struct Expression; }

BOOST_STRONG_TYPEDEF(unsigned, VariableIndex)

struct CGCheckpoint {
  int node_idx;
  int par_node_idx;
  DeviceMempoolSizes device_mem_checkpoint;
};

inline void swap(VariableIndex& i1, VariableIndex& i2) {
  VariableIndex t = i1;
  i1 = i2;
  i2 = t;
}

/**
 * \ingroup compgraph
 * \brief Computation graph where nodes represent forward and backward intermediate values, and edges represent functions of multiple values.
 * \details To represent the fact that a function may have multiple arguments, edges have a single head and 0, 1, 2, or more tails. (Constants, inputs, and parameters are represented as functions of 0 parameters.)
 * Example: given the function z = f(x, y), z, x, and y are nodes, and there is an edge representing f with which points to the z node (i.e., its head), and x and y are the tails of the edge.
 * You shouldn't need to use most methods from the ComputationGraph except for `backward` since most of them are available directly from the Expression class.
 */
struct ComputationGraph {
  /**
   * \brief Default constructor
   */
  ComputationGraph();
  ~ComputationGraph();

  // INPUTS
  /**
   * \brief Add scalar input
   * \details The computational network will pull inputs in from the user's data structures and make them available to the computation
   *
   * \param s Real number
   * \return The index of the created variable
   */
  VariableIndex add_input(real s);  //
  /**
   * \brief Add scalar input by pointer
   * \details The computational network will pull inputs in from the user's data structures and make them available to the computation
   *
   * \param ps Pointer to a real number
   * \return The index of the created variable
   */
  VariableIndex add_input(const real* ps);  // add pointer to scalar
  /**
   * \brief Add multidimentsional input
   * \details The computational network will pull inputs in from the user's data structures and make them available to the computation
   *
   * \param d Desired shape of the input
   * \param data Input data (as a 1 dimensional array)
   * \return The index of the created variable
   */
  VariableIndex add_input(const Dim& d, const std::vector<float>& data);
  /**
   * \brief Add multidimentsional input by pointer
   * \details The computational network will pull inputs in from the user's data structures and make them available to the computation
   *
   * \param d Desired shape of the input
   * \param pdata Pointer to the input data (as a 1 dimensional array)
   * \return The index of the created variable
   */
  VariableIndex add_input(const Dim& d, const std::vector<float>* pdata);
  /**
   * \brief Add sparse input
   * \details The computational network will pull inputs in from the user's data structures and make them available to the computation. Represents specified (not learned) inputs to the network in sparse array format, with an optional default value.
   *
   * \param d Desired shape of the input
   * \param ids The indexes of the data points to update
   * \param data  The data points corresponding to each index
   * \param defdata The default data with which to set the unspecified data points
   * \return The index of the created variable
   */
  VariableIndex add_input(const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

  // PARAMETERS
  // parameters are things that are optimized. in contrast to a system like
  // Torch where computational modules may have their own parameters, in DYNET
  // parameters are just parameters
  /**
   * \brief Add a parameter to the computation graph
   *
   * \param p Parameter to be added
   * \return The index of the created variable
   */
  VariableIndex add_parameters(Parameter p);
  /**
   * \brief Add a full matrix of lookup parameters to the computation graph
   *
   * \param p LookupParameter to be added
   * \return The index of the created variable
   */
  VariableIndex add_parameters(LookupParameter p);
  /**
   * \brief Add a parameter to the computation graph (but don't update)
   *
   * \param p Parameter to be added
   * \return The index of the created variable
   */
  VariableIndex add_const_parameters(Parameter p);
  /**
   * \brief Add a full matrix of lookup parameter to the computation graph (but don't update)
   *
   * \param p LookupParameter to be added
   * \return The index of the created variable
   */
  VariableIndex add_const_parameters(LookupParameter p);
  // use pindex to point to a memory location where the index will live
  // that the caller owns
  /**
   * \brief Add a lookup parameter to the computation graph
   * \details Use pindex to point to a memory location where the index will live that the caller owns
   *
   * \param p Lookup parameter from which to pick
   * \param pindex Pointer to the index to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_lookup(LookupParameter p, const unsigned* pindex);
  /**
   * \brief Add a lookup parameter to the computation graph
   *
   * \param p Lookup parameter from which to pick
   * \param index Index to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_lookup(LookupParameter p, unsigned index);
  /**
   * \brief Add lookup parameters to the computation graph
   * \details Use pindices to point to a memory location where the indices will live that the caller owns
   *
   * \param p Lookup parameter from which to pick
   * \param pindices Pointer to the indices to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  /**
   * \brief Add lookup parameters to the computation graph
   *
   * \param p Lookup parameter from which to pick
   * \param indices Indices to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>& indices);
  //
  /**
   * \brief Add a lookup parameter to the computation graph
   * \details Just like add_lookup, but don't optimize the lookup parameters
   *
   * \param p Lookup parameter from which to pick
   * \param pindex Pointer to the indices to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_const_lookup(LookupParameter p, const unsigned* pindex);
  /**
   * \brief Add a lookup parameter to the computation graph
   * \details Just like add_lookup, but don't optimize the lookup parameters
   *
   * \param p Lookup parameter from which to pick
   * \param index Index to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_const_lookup(LookupParameter p, unsigned index);
  /**
   * \brief Add lookup parameters to the computation graph
   * \details Just like add_lookup, but don't optimize the lookup parameters
   *
   * \param p Lookup parameter from which to pick
   * \param pindices Pointer to the indices to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  /**
   * \brief Add lookup parameters to the computation graph
   * \details Just like add_lookup, but don't optimize the lookup parameters
   *
   * \param p Lookup parameter from which to pick
   * \param indices Indices to lookup
   *
   * \return The index of the created variable
   */
  VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>& indices);

  // COMPUTATIONS
  /**
   * \brief Add a function to the computation graph
   * \details This what is called when creating an expression
   *
   * \param arguments List of the arguments indices
   * \tparam Function Function to be applied
   * \return The index of the output variable
   */
  template <class Function> inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments);
  /**
   * \brief Add a function to the computation graph (with side information)
   * \details This what is called when creating an expression
   *
   * \param arguments List of the arguments indices
   * \param side_information Side information that is needed to compute the function
   * \tparam Function Function to be applied
   * \return The index of the output variable
   */
  template <class Function, typename... Args>
  inline VariableIndex add_function(const std::initializer_list<VariableIndex>& arguments,
                                    Args&&... side_information);
  template <class Function, typename T>
  inline VariableIndex add_function(const T& arguments);
  template <class Function, typename T, typename... Args>
  inline VariableIndex add_function(const T& arguments,
                                    Args&&... side_information);

  // reset ComputationGraph to a newly created state
  /**
   * \brief Reset ComputationGraph to a newly created state
   * \details [long description]
   */
  void clear();
  /**
   * \brief Set a checkpoint
   */
  void checkpoint();
  /**
   * \brief Revert to last checkpoint
   */
  void revert();

  /**
   * \brief Get dimension of a node
   *
   * \param index Variable index of the node
   * \return Dimension
   */
  Dim& get_dimension(VariableIndex index) const;


  // perform computations

  // run complete forward pass from first node to given one, ignoring all precomputed values.
  /**
   * \brief Run complete forward pass from first node to given one, ignoring all precomputed values.
   *
   * \param last Expression up to which the forward pass must be computed
   * \return Value of the `last` Expression after execution
   */
  const Tensor& forward(const expr::Expression& last);
  /**
   * \brief Run complete forward pass from first node to given one, ignoring all precomputed values.
   *
   * \param i Variable index of the node up to which the forward pass must be computed
   * \return Value of the end Node after execution
   */
  const Tensor& forward(VariableIndex i);
  /**
   * \brief Run forward pass from the last computed node to given one.
   * \details Useful if you want to add nodes and evaluate just the new parts.
   *
   * \param last Expression up to which the forward pass must be computed
   * \return Value of the `last` Expression after execution
   */
  const Tensor& incremental_forward(const expr::Expression& last);
  /**
   * \brief Run forward pass from the last computed node to given one.
   * \details Useful if you want to add nodes and evaluate just the new parts.
   *
   * \param last Variable index of the node up to which the forward pass must be computed
   * \return Value of the end Node after execution
   */
  const Tensor& incremental_forward(VariableIndex i);
  /**
   * \brief Get forward value for node at index i.
   * \details Performs forward evaluation if note available (may compute more than strictly what is needed).
   *
   * \param i Index of the variable from which you want the value
   * \return Requested value
   */
  const Tensor& get_value(VariableIndex i);
  /**
   * \brief Get forward value for the given expression
   * \details Performs forward evaluation if note available (may compute more than strictly what is needed).
   *
   * \param e Expression from which you want the value
   * \return Requested value
   */
  const Tensor& get_value(const expr::Expression& e);

  /**
   * \brief Get gradient for node at index i.
   * \details Performs backward pass if not available (may compute more than strictly what is needed).
   *
   * \param i Index of the variable from which you want the gradient
   * \return Requested gradient
   */
  const Tensor& get_gradient(VariableIndex i);
  /**
   * \brief Get forward gradient for the given expression
   * \details Performs backward pass if not available (may compute more than strictly what is needed).
   *
   * \param e Expression from which you want the gradient
   * \return Requested gradient
   */
  const Tensor& get_gradient(const expr::Expression& e);
  /**
   * \brief Clears forward caches (for get_value etc).
   */
  void invalidate();
  /**
   * \brief Computes backward gradients from the front-most evaluated node.
   * 
   * \details The parameter `full` specifies whether the gradients should be computed for all nodes (`true`) or only non-constant nodes.
   * 
   * By default, a node is constant unless
   * 
   * 1. it is a parameter node
   * 2. it depends on a non-constant node
   * 
   * Thus, functions of constants and inputs are considered as constants.
   * 
   * Turn `full` on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.
   *
   * \param last Expression from which to compute the gradient
   * \param full Whether to compute all gradients (including with respect to constant nodes). 
   */
  void backward(const expr::Expression& last, bool full = false);
  /**
   * \brief Computes backward gradients from node i (assuming it already been evaluated).
   * 
   * \details The parameter `full` specifies whether the gradients should be computed for all nodes (`true`) or only non-constant nodes.
   * 
   * By default, a node is constant unless
   * 
   * 1. it is a parameter node
   * 2. it depends on a non-constant node
   * 
   * Thus, functions of constants and inputs are considered as constants.
   * 
   * Turn `full` on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.
   *
   * \param i Index of the node from which to compute the gradient
   * \param full Whether to compute all gradients (including with respect to constant nodes). Turn this on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.
   */
  void backward(VariableIndex i, bool full = false);
  // set immediate_compute variable
  void set_immediate_compute(bool ic);
  // set check_validity variable
  void set_check_validity(bool cv);

  /**
   * \brief Used for debugging
   */
  void print_graphviz() const;

  /**
   * \brief Get the unique graph ID
   * \details This ID is incremented by 1 each time a computation graph is created
   * \return graph is
   */
  unsigned get_id() const {return graph_id;};

  // data
  std::vector<Node*> nodes;       // **stored in topological order**
  std::vector<VariableIndex> parameter_nodes; // nodes that contain parameters that can be updated (subset of nodes)

  ExecutionEngine* ee;  // handles the execution
private:
  unsigned graph_id;
  // flag of whether to compute immediately for each expression, i.e., an imperative execution style to help debug.
  bool immediate_compute;
  // flag of checking Inf/NaN of each layer. Only performing checking when immediate_compute is also set to true.
  bool check_validity;
  void set_dim_for_new_node(const VariableIndex& i);

  std::vector<CGCheckpoint> checkpoints;
  CGCheckpoint _get_checkpoint();
  void _revert(CGCheckpoint checkpoint);
};

// represents an SSA variable
// * in_edge is the **ordered** list of indices of the function arguments
// * fx is the computed value of the variable
// * dEdf is the derivative of the output with respect to the function
/**
 * \ingroup nodes
 * \brief Represents an SSA variable
 * \details Contains information on tha computation node : arguments, output value and gradient of the output with respect to the function.
 * This class must be inherited to implement any new operation. See nodes.cc for examples.
 * An operation on expressions can then be created from the new Node, see expr.h/expr.cc for examples
 */
struct Node {
  virtual ~Node();

  /**
   * \brief Compute dimensions of result for given dimensions of inputs
   * \details Also checks to make sure inputs are compatible with each other
   *
   * \param xs Vector containing the dimensions of the inputs
   * \return Dimension of the output
   */
  virtual Dim dim_forward(const std::vector<Dim>& xs) const = 0;

  // for debugging
  /**
   * \brief Returns important information for debugging
   * \details See nodes-conv.cc for examples
   *
   * \param args String descriptions of the arguments
   * \return String description of the node
   */
  virtual std::string as_string(const std::vector<std::string>& args) const = 0;

  // in general, this will return an empty size, but if a component needs to store
  // extra information in the forward pass for use in the backward pass, it can
  // request the memory here (nb. you could put it on the Node object, but in general,
  // edges should not allocate tensor memory since memory is managed centrally for the
  // entire computation graph).
  /**
   * \brief Size of the auxiliar storage
   * \details in general, this will return an empty size, but if a component needs to store extra information in the forward pass for use in the backward pass, it can request the memory here (nb. you could put it on the Node object, but in general, edges should not allocate tensor memory since memory is managed centrally for the entire computation graph).
   * \return Size
   */
  virtual size_t aux_storage_size() const;


  // computation
  /**
   * \brief Forward computation
   * \details This function contains the logic for the forward pass. Some implementation remarks from nodes.cc:
   * 1. fx can be understood as a pointer to the (preallocated) location for the result of forward to be stored
   * 2. fx is not initialized, so after calling forward fx must point to the correct answer
   * 3. fx can be repointed to an input, if forward(x) evaluates to x (e.g., in reshaping)
   * 4. scalars results of forward are placed in fx.v[0]
   * 5. DYNET manages its own memory, not Eigen, and it is configured with the EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate memory, it is (probably) because of an implicit creation of a temporary variable. To tell Eigen this is not necessary, the noalias() method is available. If you really do need a temporary variable, its capacity must be requested by Node::aux_storage_size
   *
   * Note on debugging problems with differentiable components
   *
   * - fx is uninitialized when forward is called- are you relying on it being 0?
   *
   * \param xs Pointers to the inputs
   * \param fx pointer to the (preallocated) location for the result of forward to be stored
   */
  virtual void forward_impl(const std::vector<const Tensor*>& xs,
                            Tensor& fx) const = 0;
  //
  /**
   * \brief Accumulates the derivative of E with respect to the ith argument to f, that is, xs[i]
   * \details This function contains the logic for the backward pass. Some implementation remarks from nodes.cc:
   * 1. dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on the same x_i. Even, e.g., Identity must be implemented as dEdx1 += dEdf. THIS IS EXTREMELY IMPORTANT
   * 2. scalars results of forward are placed in fx.v[0]
   * 3. DYNET manages its own memory, not Eigen, and it is configured with the EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate memory, it is (probably) because of an implicit creation of a temporary variable. To tell Eigen this is not necessary, the noalias() method is available. If you really do need a temporary variable, its capacity must be requested by Node::aux_storage_size
   *
   * Note on debugging problems with differentiable components
   *
   * - dEdxi must accummulate (see point 4 above!)
   *
   * \param xs Pointers to inputs
   * \param fx Output
   * \param dEdf Gradient of the objective w.r.t the output of the node
   * \param i Index of the input w.r.t which we take the derivative
   * \param dEdxi Gradient of the objective w.r.t the input of the node
   */
  virtual void backward_impl(const std::vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const = 0;

  /**
   * \brief Whether this node supports computing multiple batches in one call.
   * \details If true, forward and backward will be called once with a multi-batch tensor. If false, forward and backward will be called multiple times for each item.
   * \return Support for multibatch
   */
  virtual bool supports_multibatch() const { return false; }

  // perform the forward/backward passes in one or multiple calls
  /**
   * \brief perform the forward/backward passes in one or multiple calls
   *
   * \param xs Pointers to the inputs
   * \param fx pointer to the (preallocated) location for the result of forward to be stored
   */
  virtual void forward(const std::vector<const Tensor*>& xs,
                       Tensor& fx) const final;
  /**
   * \brief perform the backward passes in one or multiple calls
   *
   * \param xs Pointers to inputs
   * \param fx Output
   * \param dEdf Gradient of the objective w.r.t the output of the node
   * \param i Index of the input w.r.t which we take the derivative
   * \param dEdxi Gradient of the objective w.r.t the input of the node
   */
  virtual void backward(const std::vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const final;

  //
  /**
   * \brief Number of arguments to the function
   * \return Arity of the function
   */
  inline unsigned arity() const { return args.size(); }

  inline void set_cg(ComputationGraph* cg) { cg_ = cg; }

  inline ComputationGraph* get_cg() const {
    if (cg_) return cg_;
    else return NULL;
  }

  std::vector<VariableIndex> args;/**< Dependency structure */

  // memory size
  Dim dim; /**< Will be .size() = 0 initially filled in by forward() -- TODO fix this */

  Device* device; /**< pointer to the node, or null to inherit device from first input, or default when there is no input */

protected:
  Node() : args(), device(default_device) {}
  explicit Node(const std::initializer_list<VariableIndex>& a) : args(a), device(default_device) {}
  template <typename T>
  explicit Node(const T&c) : args(c.begin(), c.end()), device(default_device) {}

private:
  ComputationGraph* cg_;  // pointer to the computation graph

public:
  // auxiliary memory
  mutable void* aux_mem; /**< this will usually be null. but, if your node needs to store intermediate values between forward and backward, you can use store it here. request the number of bytes you need from aux_storage_size(). Note: this memory will be on the CPU or GPU, depending on your computation backend*/
};

template <class Function>
inline VariableIndex ComputationGraph::add_function(const std::initializer_list<VariableIndex>& arguments) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

// pass side information to the function. these are likely to be nondifferentiable arguments
template <class Function, typename... Args>
inline VariableIndex ComputationGraph::add_function(const std::initializer_list<VariableIndex>& arguments,
    Args&&... side_information) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments, std::forward<Args>(side_information)...));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

template <class Function, typename T>
inline VariableIndex ComputationGraph::add_function(const T& arguments) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

// pass side information to the function. these are likely to be nondifferentiable arguments
template <class Function, typename T, typename... Args>
inline VariableIndex ComputationGraph::add_function(const T& arguments,
    Args&&... side_information) {
  VariableIndex new_node_index(nodes.size());
  nodes.push_back(new Function(arguments, std::forward<Args>(side_information)...));
  set_dim_for_new_node(new_node_index);
  return new_node_index;
}

} // namespace dynet

#endif
