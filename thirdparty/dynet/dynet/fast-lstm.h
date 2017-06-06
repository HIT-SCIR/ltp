#ifndef DYNET_FAST_LSTM_H_
#define DYNET_FAST_LSTM_H_

#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"

using namespace dynet::expr;

namespace dynet {

class Model;

/*
FastLSTM replaces the matrices from cell to other units, by diagonal matrices.
*/
struct FastLSTMBuilder : public RNNBuilder {
  FastLSTMBuilder() = default;
  explicit FastLSTMBuilder(unsigned layers,
                           unsigned input_dim,
                           unsigned hidden_dim,
                           Model& model);

  Expression back() const override { return (cur == -1? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    for(auto my_h : final_h()) ret.push_back(my_h);
    return ret;
  }
  unsigned num_h0_components() const override { return 2 * layers; }

  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const override {
    std::vector<Expression> ret = (i == -1 ? c0 : c[i]);
    for(auto my_h : get_h(i)) ret.push_back(my_h);
    return ret;
  }

  void copy(const RNNBuilder & params) override;
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override;
  Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override;

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
};

} // namespace dynet

#endif
