#include "dynet/gru.h"

#include <string>
#include <vector>
#include <iostream>

#include "dynet/nodes.h"
#include "dynet/training.h"

using namespace std;

namespace dynet {

enum { X2Z, H2Z, BZ, X2R, H2R, BR, X2H, H2H, BH };

GRUBuilder::GRUBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model& model) : hidden_dim(hidden_dim), layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // z
    Parameter p_x2z = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2z = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bz = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // r
    Parameter p_x2r = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2r = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_br = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // h
    Parameter p_x2h = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2h = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bh = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2z, p_h2z, p_bz, p_x2r, p_h2r, p_br, p_x2h, p_h2h, p_bh};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;
}

void GRUBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    // z
    Expression x2z = parameter(cg, p[X2Z]);
    Expression h2z = parameter(cg, p[H2Z]);
    Expression bz = parameter(cg, p[BZ]);

    // r
    Expression x2r = parameter(cg, p[X2R]);
    Expression h2r = parameter(cg, p[H2R]);
    Expression br = parameter(cg, p[BR]);

    // h
    Expression x2h = parameter(cg, p[X2H]);
    Expression h2h = parameter(cg, p[H2H]);
    Expression bh = parameter(cg, p[BH]);

    vector<Expression> vars = {x2z, h2z, bz, x2r, h2r, br, x2h, h2h, bh};
    param_vars.push_back(vars);
  }
}

void GRUBuilder::start_new_sequence_impl(const std::vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  DYNET_ARG_CHECK(h0.empty() || h0.size() == layers,
                          "Number of inputs passed to initialize GRUBuilder (" << h0.size() << ") "
                          "is not equal to the number of layers (" << layers << ")");
}

Expression GRUBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "Number of inputs passed to RNNBuilder::set_h() (" << h_new.size() << ") "
                          "is not equal to the number of layers (" << layers << ")");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    h[t][i] = h_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression GRUBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  return set_h_impl(prev, s_new);
}

Expression GRUBuilder::add_input_impl(int prev, const Expression& x) {
  //if(dropout_rate != 0.f)
  //throw std::runtime_error("GRUBuilder doesn't support dropout yet");
  const bool has_initial_state = (h0.size() > 0);
  h.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression h_tprev;
    // prev_zero means that h_tprev should be treated as 0
    bool prev_zero = false;
    if (prev >= 0 || has_initial_state) {
      h_tprev = (prev < 0) ? h0[i] : h[prev][i];
    } else { prev_zero = true; }
    if (dropout_rate) in = dropout(in, dropout_rate);
    // update gate
    Expression zt;
    if (prev_zero)
      zt = affine_transform({vars[BZ], vars[X2Z], in});
    else
      zt = affine_transform({vars[BZ], vars[X2Z], in, vars[H2Z], h_tprev});
    zt = logistic(zt);
    // forget
    Expression ft = 1.f - zt;
    // reset gate
    Expression rt;
    if (prev_zero)
      rt = affine_transform({vars[BR], vars[X2R], in});
    else
      rt = affine_transform({vars[BR], vars[X2R], in, vars[H2R], h_tprev});
    rt = logistic(rt);

    // candidate activation
    Expression ct;
    if (prev_zero) {
      ct = affine_transform({vars[BH], vars[X2H], in});
      ct = tanh(ct);
      Expression nwt = cmult(zt, ct);
      in = ht[i] = nwt;
    } else {
      Expression ght = cmult(rt, h_tprev);
      ct = affine_transform({vars[BH], vars[X2H], in, vars[H2H], ght});
      ct = tanh(ct);
      Expression nwt = cmult(zt, ct);
      Expression crt = cmult(ft, h_tprev);
      in = ht[i] = crt + nwt;
    }
  }
  if (dropout_rate) return dropout(ht.back(), dropout_rate);
  else return ht.back();
}

void GRUBuilder::copy(const RNNBuilder & rnn) {
  const GRUBuilder & rnn_gru = (const GRUBuilder&)rnn;
  if(params.size() != rnn_gru.params.size())
    DYNET_INVALID_ARG("Attempt to copy between two GRUBuilders that are not the same size");
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_gru.params[i][j];
}

} // namespace dynet
