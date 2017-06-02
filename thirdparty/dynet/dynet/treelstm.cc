#include <string>
#include <vector>
#include <iostream>

#include "dynet/nodes.h"
#include "dynet/treelstm.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

BOOST_CLASS_EXPORT_IMPLEMENT(TreeLSTMBuilder)
BOOST_CLASS_EXPORT_IMPLEMENT(NaryTreeLSTMBuilder)
BOOST_CLASS_EXPORT_IMPLEMENT(UnidirectionalTreeLSTMBuilder)
BOOST_CLASS_EXPORT_IMPLEMENT(BidirectionalTreeLSTMBuilder)

enum { X2I, BI, X2F, BF, X2O, BO, X2C, BC };
enum { H2I, H2F, H2O, H2C, C2I, C2F, C2O };

Expression TreeLSTMBuilder::add_input_impl(int prev, const Expression& x) { throw std::runtime_error("add_input_impl() not a valid function for TreeLSTMBuilder"); }
Expression TreeLSTMBuilder::back() const { throw std::runtime_error("back() not a valid function for TreeLSTMBuilder"); }
std::vector<Expression> TreeLSTMBuilder::final_h() const { throw std::runtime_error("final_h() not a valid function for TreeLSTMBuilder"); }
std::vector<Expression> TreeLSTMBuilder::final_s() const { throw std::runtime_error("final_s() not a valid function for TreeLSTMBuilder"); }
unsigned TreeLSTMBuilder::num_h0_components() const { throw std::runtime_error("num_h0_components() not a valid function for TreeLSTMBuilder"); }
void TreeLSTMBuilder::copy(const RNNBuilder&) { throw std::runtime_error("copy() not a valid function for TreeLSTMBuilder"); }

DYNET_SERIALIZE_COMMIT(TreeLSTMBuilder, DYNET_SERIALIZE_DERIVED_EQ_DEFINE(RNNBuilder))
DYNET_SERIALIZE_IMPL(TreeLSTMBuilder);

// See "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
// by Tai, Nary, and Manning (2015), section 3.2, for details on this model.
// http://arxiv.org/pdf/1503.00075v3.pdf
NaryTreeLSTMBuilder::NaryTreeLSTMBuilder(unsigned N,
                         unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model& model) : layers(layers), N(N), cg(nullptr) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = model.add_parameters({hidden_dim, layer_input_dim});
    LookupParameter p_h2i = model.add_lookup_parameters(N, {hidden_dim, hidden_dim});
    LookupParameter p_c2i = model.add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameter p_bi = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // f
    Parameter p_x2f = model.add_parameters({hidden_dim, layer_input_dim});
    LookupParameter p_h2f = model.add_lookup_parameters(N*N, {hidden_dim, hidden_dim});
    LookupParameter p_c2f = model.add_lookup_parameters(N*N, {hidden_dim, hidden_dim});
    Parameter p_bf = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // o
    Parameter p_x2o = model.add_parameters({hidden_dim, layer_input_dim});
    LookupParameter p_h2o = model.add_lookup_parameters(N, {hidden_dim, hidden_dim});
    LookupParameter p_c2o = model.add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameter p_bo = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // c (a.k.a. u)
    Parameter p_x2c = model.add_parameters({hidden_dim, layer_input_dim});
    LookupParameter p_h2c = model.add_lookup_parameters(N, {hidden_dim, hidden_dim});
    Parameter p_bc = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_bi, p_x2f, p_bf, p_x2o, p_bo, p_x2c, p_bc};
    vector<LookupParameter> lps = {p_h2i, p_h2f, p_h2o, p_h2c, p_c2i, p_c2f, p_c2o};
    params.push_back(ps);
    lparams.push_back(lps);
  }  // layers
}

void NaryTreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
  this->cg = &cg;
  param_vars.clear();
  lparam_vars.clear();
  param_vars.reserve(layers);
  lparam_vars.reserve(layers);

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];
    auto& lp = lparams[i];

    //i
    Expression i_x2i = parameter(cg, p[X2I]);
    Expression i_bi = parameter(cg, p[BI]);
    //f
    Expression i_x2f = parameter(cg, p[X2F]);
    Expression i_bf = parameter(cg, p[BF]);
    //o
    Expression i_x2o = parameter(cg, p[X2O]);
    Expression i_bo = parameter(cg, p[BO]);
    //c
    Expression i_x2c = parameter(cg, p[X2C]);
    Expression i_bc = parameter(cg, p[BC]);

    vector<Expression> vars = {i_x2i, i_bi, i_x2f, i_bf, i_x2o, i_bo, i_x2c, i_bc};
    param_vars.push_back(vars);

    DYNET_ASSERT(lp.size() == C2O + 1, "Dimension mismatch in TreeLSTM");
    vector<vector<Expression>> lvars(lp.size());
    for (unsigned p_type = H2I; p_type <= C2O; p_type++) {
    LookupParameter p = lp[p_type];
      vector<Expression> vals(p.get()->values.size());
      for (unsigned k = 0; k < p.get()->values.size(); ++k) {
        //vals[k] = lookup(cg, p, k);
        vals[k].i = 0;
      }
      lvars[p_type] = vals;
    }
    lparam_vars.push_back(lvars);
  }
}

Expression NaryTreeLSTMBuilder::Lookup(unsigned layer, unsigned p_type, unsigned value) {
  if (lparam_vars[layer][p_type][value].i == 0) {
    LookupParameter p = lparams[layer][p_type];
    lparam_vars[layer][p_type][value] = lookup(*cg, p, value);
  }
  return lparam_vars[layer][p_type][value];
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void NaryTreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers*2 == hinit.size(),
                            "Incorrectly sized initialization in TreeLSTM (" << hinit.size() << "). "
                            "Must be twice the number of layers (which is " << layers<< ")");
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

Expression NaryTreeLSTMBuilder::add_input(int id, vector<int> children, const Expression& x) {
  DYNET_ASSERT(id >= 0 && h.size() == (unsigned)id, "Failed dimension check in TreeLSTMBuilder");
  DYNET_ASSERT(id >= 0 && c.size() == (unsigned)id, "Failed dimension check in TreeLSTMBuilder");
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();

  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    vector<Expression> i_h_children, i_c_children;
    i_h_children.reserve(children.size() > 1 ? children.size() : 1);
    i_c_children.reserve(children.size() > 1 ? children.size() : 1);

    bool has_prev_state = (children.size() > 0 || has_initial_state);
    if (children.size() == 0) {
      i_h_children.push_back(Expression());
      i_c_children.push_back(Expression());
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_children[0] = h0[i];
        i_c_children[0] = c0[i];
      }
    }
    else {  // t > 0
      for (int child : children) {
        i_h_children.push_back(h[child][i]);
        i_c_children.push_back(c[child][i]);
      }
    }

    // input
    Expression i_ait;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BI], vars[X2I], in};
      xs.reserve(4 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(Lookup(i, H2I, ej));
        xs.push_back(i_h_children[j]);
        xs.push_back(Lookup(i, C2I, ej));
        xs.push_back(i_c_children[j]);
      }
      DYNET_ASSERT(xs.size() == 4 * children.size() + 3, "Failed dimension check in TreeLSTMBuilder");
      i_ait = affine_transform(xs);
    }
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);

    // forget
    vector<Expression> i_ft;
    for (unsigned k = 0; k < children.size(); ++k) {
      unsigned ek = (k < N) ? k : N - 1;
      Expression i_aft;
      if (has_prev_state) {
        vector<Expression> xs = {vars[BF], vars[X2F], in};
        xs.reserve(4 * children.size() + 3);
        for (unsigned j = 0; j < children.size(); ++j) {
          unsigned ej = (j < N) ? j : N - 1;
          xs.push_back(Lookup(i, H2F, ej * N + ek));
          xs.push_back(i_h_children[j]);
          xs.push_back(Lookup(i, C2F, ej * N + ek));
          xs.push_back(i_c_children[j]);
        }
        DYNET_ASSERT(xs.size() == 4 * children.size() + 3, "Failed dimension check in TreeLSTMBuilder");
        i_aft = affine_transform(xs);
      }
      else
        i_aft = affine_transform({vars[BF], vars[X2F], in});
      i_ft.push_back(logistic(i_aft + 1.f));
    }

    // write memory cell
    Expression i_awt;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BC], vars[X2C], in};
      // This is the one and only place that should *not* condition on i_c_children
      // This should condition only on x (a.k.a. in), the bias (vars[BC]) and i_h_children
      xs.reserve(2 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(Lookup(i, H2C, ej));
        xs.push_back(i_h_children[j]);
      }
      DYNET_ASSERT(xs.size() == 2 * children.size() + 3, "Failed dimension check in TreeLSTMBuilder");
      i_awt = affine_transform(xs);
    }
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);

    // compute new cell value
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it, i_wt);
      vector<Expression> i_crts(children.size());
      for (unsigned j = 0; j < children.size(); ++j) {
        i_crts[j] = cmult(i_ft[j], i_c_children[j]);
      }
      Expression i_crt = sum(i_crts);
      ct[i] = i_crt + i_nwt;
    }
    else {
      ct[i] = cmult(i_it, i_wt);
    }

    // output
    Expression i_aot;
    if (has_prev_state) {
      vector<Expression> xs = {vars[BO], vars[X2O], in};
      xs.reserve(4 * children.size() + 3);
      for (unsigned j = 0; j < children.size(); ++j) {
        unsigned ej = (j < N) ? j : N - 1;
        xs.push_back(Lookup(i, H2O, ej));
        xs.push_back(i_h_children[j]);
        xs.push_back(Lookup(i, C2O, ej));
        xs.push_back(i_c_children[j]);
      }
      DYNET_ASSERT(xs.size() == 4 * children.size() + 3, "Failed dimension check in TreeLSTMBuilder");
      i_aot = affine_transform(xs);
    }
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);

    // Compute new h value
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot, ph_t);
  }
  return ht.back();
}

void NaryTreeLSTMBuilder::copy(const RNNBuilder & rnn) {
  const NaryTreeLSTMBuilder & rnn_treelstm = (const NaryTreeLSTMBuilder&)rnn;
  DYNET_ASSERT(params.size() == rnn_treelstm.params.size(), "Failed dimension check in TreeLSTMBuilder");
  for(size_t i = 0; i < params.size(); ++i) {
    for(size_t j = 0; j < params[i].size(); ++j) {
      params[i][j] = rnn_treelstm.params[i][j];
    }
  }
}

DYNET_SERIALIZE_COMMIT(NaryTreeLSTMBuilder, DYNET_SERIALIZE_DERIVED_DEFINE(TreeLSTMBuilder, params, lparams, layers, N))
DYNET_SERIALIZE_IMPL(NaryTreeLSTMBuilder);

UnidirectionalTreeLSTMBuilder::UnidirectionalTreeLSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model& model) {
  node_builder = LSTMBuilder(layers, input_dim, hidden_dim, model);
}

void UnidirectionalTreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
  node_builder.new_graph(cg);
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void UnidirectionalTreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  node_builder.start_new_sequence(hinit);
}

Expression UnidirectionalTreeLSTMBuilder::add_input(int id, vector<int> children, const Expression& x) {
  DYNET_ASSERT(id >= 0 && h.size() == (unsigned)id, "Failed dimension check in TreeLSTMBuilder");

  RNNPointer prev = (RNNPointer)(-1);
  Expression embedding = node_builder.add_input(prev, x);
  prev = node_builder.state();

  for (unsigned child : children) {
    embedding = node_builder.add_input(prev, h[child]);
    prev = node_builder.state();
  }
  h.push_back(embedding);
  return embedding;
}

DYNET_SERIALIZE_COMMIT(UnidirectionalTreeLSTMBuilder, DYNET_SERIALIZE_DERIVED_DEFINE(TreeLSTMBuilder, node_builder))
DYNET_SERIALIZE_IMPL(UnidirectionalTreeLSTMBuilder);

BidirectionalTreeLSTMBuilder::BidirectionalTreeLSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model& model) {
  DYNET_ASSERT(hidden_dim % 2 == 0, "Failed dimension check in TreeLSTMBuilder");
  fwd_node_builder = LSTMBuilder(layers, input_dim, hidden_dim / 2, model);
  rev_node_builder = LSTMBuilder(layers, input_dim, hidden_dim / 2, model);
}

void BidirectionalTreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {
  fwd_node_builder.new_graph(cg);
  rev_node_builder.new_graph(cg);
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void BidirectionalTreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  fwd_node_builder.start_new_sequence(hinit);
  rev_node_builder.start_new_sequence(hinit);
}

Expression BidirectionalTreeLSTMBuilder::add_input(int id, vector<int> children, const Expression& x) {
  DYNET_ASSERT(id >= 0 && h.size() == (unsigned)id, "Failed dimension check in TreeLSTMBuilder");

  RNNPointer prev = (RNNPointer)(-1);
  Expression fwd_embedding = fwd_node_builder.add_input(prev, x);
  prev = fwd_node_builder.state();
  for (unsigned child : children) {
    fwd_embedding = fwd_node_builder.add_input(prev, h[child]);
    prev = fwd_node_builder.state();
  }

  prev = (RNNPointer)(-1);
  Expression rev_embedding = rev_node_builder.add_input(prev, x);
  prev = rev_node_builder.state();
  for (unsigned i = children.size(); i-- > 0;) {
    unsigned  child = children[i];
    rev_embedding = rev_node_builder.add_input(prev, h[child]);
    prev = rev_node_builder.state();
  }

  Expression embedding = concatenate({fwd_embedding, rev_embedding});
  h.push_back(embedding);

  return embedding;
}

Expression BidirectionalTreeLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) { throw std::runtime_error("set_h() not a valid function for BidirectionalTreeLSTMBuilder"); }

DYNET_SERIALIZE_COMMIT(BidirectionalTreeLSTMBuilder, DYNET_SERIALIZE_DERIVED_DEFINE(TreeLSTMBuilder, fwd_node_builder, rev_node_builder))
DYNET_SERIALIZE_IMPL(BidirectionalTreeLSTMBuilder);
