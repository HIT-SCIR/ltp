#include "dynet/deep-lstm.h"

#include <string>
#include <vector>
#include <iostream>

#include "dynet/nodes.h"

using namespace std;
using namespace dynet::expr;

namespace dynet {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

DeepLSTMBuilder::DeepLSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model& model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // o
    Parameter p_x2o = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2o = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2o = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // c
    Parameter p_x2c = model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2c = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim + input_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
}

void DeepLSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_c2i = parameter(cg,p[C2I]);
    Expression i_bi = parameter(cg,p[BI]);
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_c2o = parameter(cg,p[C2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void DeepLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                            "DeepLSTMBuilder must be initialized with 2 times as many expressions as layers "
                            "(hidden state and cell for each layer). However, for " << layers << " layers, "
                            << hinit.size() << " expressions were passed in");
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression DeepLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  o.push_back(Expression());
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression& ot = o.back();
  Expression in = x;
  vector<Expression> cc(layers);
  for (unsigned i = 0; i < layers; ++i) {
    if (i > 0)
      in = concatenate({in, x});
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    // input
    Expression i_ait;
    if (has_prev_state)
//      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
//      i_ait = vars[BI] + vars[X2I] * in;
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
    Expression i_ft = 1.f - i_it;
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
//      i_awt = vars[BC] + vars[X2C] * in + vars[H2C]*i_h_tm1;
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
//      i_awt = vars[BC] + vars[X2C] * in;
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it,i_wt);
      Expression i_crt = cmult(i_ft,i_c_tm1);
      ct[i] = i_crt + i_nwt;
    } else {
      ct[i] = cmult(i_it,i_wt);
    }

    Expression i_aot;
    if (has_prev_state)
//      i_aot = vars[BO] + vars[X2O] * in + vars[H2O] * i_h_tm1 + vars[C2O] * ct[i];
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
//      i_aot = vars[BO] + vars[X2O] * in;
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot,ph_t);
    cc[i] = in;
  }
  ot = concatenate(cc);
  return ot;
}

} // namespace dynet
