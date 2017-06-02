/**
 * \file lstm.h
 * \brief Helper structures to build recurrent units
 *
 * \details TODO: Create documentation and explain rnns, etc...
 */
#ifndef DYNET_LSTM_H_
#define DYNET_LSTM_H_

#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"
#include <boost/serialization/version.hpp>


using namespace dynet::expr;

namespace dynet {

class Model;
/**
 * \ingroup rnnbuilders
 * \brief LSTMBuilder creates an LSTM unit with coupled input and forget gate as well as peepholes connections.
 *
 * \details More specifically, here are the equations for the dynamics of this cell :
 *
 * \f$
 * \begin{split}
    i_t & =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)\\
    \tilde{c_t} & = \tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)\\
    c_t & = c_{t-1}\circ (1-i_t) + \tilde{c_t}\circ i_t\\
     & = c_{t-1} + (\tilde{c_t}-c_{t-1})\circ i_t\\
    o_t & = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+W_{oc}c_{t}+b_o)\\
    h_t & = \tanh(c_t)\circ o_t\\
\end{split}
\f$
 */
struct LSTMBuilder : public RNNBuilder {
  /**
   * \brief Default constructor
   */
  LSTMBuilder() = default;
  /**
   * \brief Constructor for the LSTMBuilder
   *
   * \param layers Number of layers
   * \param input_dim Dimention of the input \f$x_t\f$
   * \param hidden_dim Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$
   * \param model Model holding the parameters
   */
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model& model);

  Expression back() const override { return (cur == -1 ? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    for (auto my_h : final_h()) ret.push_back(my_h);
    return ret;
  }
  /**
   * @brief Number of components in `h_0`
   * @details For `LSTMBuilder`, this corresponds to `2 * layers` because it includes the initial cell state \f$c_0\f$
   * @return `2 * layers`
   */
  unsigned num_h0_components() const override { return 2 * layers; }

  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }

  /**
   * @brief Get the final state of the hidden layer
   * @details For `LSTMBuilder`, this consists of a vector of the memory cell values for each layer (l1, l2, l3),
   *          followed by the hidden state values
   * @return {c_{l1}, c_{l1}, ..., h_{l1}, h_{l2}, ...}
   */
  std::vector<Expression> get_s(RNNPointer i) const override {
    std::vector<Expression> ret = (i == -1 ? c0 : c[i]);
    for (auto my_h : get_h(i)) ret.push_back(my_h);
    return ret;
  }

  void copy(const RNNBuilder & params) override;

  void save_parameters_pretraining(const std::string& fname) const override;
  void load_parameters_pretraining(const std::string& fname) override;
  /**
   * \brief Set the dropout rates to a unique value
   * \details This has the same effect as `set_dropout(d,d_h,d_c)` except that all the dropout rates are set to the same value.
   * \param d Dropout rate to be applied on all of \f$x,h,c\f$
   */
  void set_dropout(float d);
  /**
   * \brief Set the dropout rates
   * \details The dropout implemented here is an adaptation of the variational dropout with tied weights introduced in [Gal, 2016](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
   * More specifically, dropout masks \f$\mathbf{z_x}\sim \mathrm{Bernoulli}(1-d_x)\f$,\f$\mathbf{z_h}\sim \mathrm{Bernoulli}(1-d_h)\f$,\f$\mathbf{z_c}\sim \mathrm{Bernoulli}(1-d_c)\f$ are sampled at the start of each sequence.
   * The dynamics of the cell are then modified to :
   *
   * \f$
   * \begin{split}
    i_t & =\sigma(W_{ix}(\frac 1 {1-d_x} {\mathbf{z_x}} \circ x_t)+W_{ih}(\frac 1 {1-d_h} {\mathbf{z_h}} \circ h_{t-1})+W_{ic}(\frac 1 {1-d_c} {\mathbf{z_c}} \circ c_{t-1})+b_i)\\
    \tilde{c_t} & = \tanh(W_{cx}(\frac 1 {1-d_x} {\mathbf{z_x}} \circ x_t)+W_{ch}(\frac 1 {1-d_h} {\mathbf{z_h}} \circ h_{t-1})+b_c)\\
    c_t & = c_{t-1}\circ (1-i_t) + \tilde{c_t}\circ i_t\\
     & = c_{t-1} + (\tilde{c_t}-c_{t-1})\circ i_t\\
    o_t & = \sigma(W_{ox}(\frac 1 {1-d_x} {\mathbf{z_x}} \circ x_t)+W_{oh}(\frac 1 {1-d_h} {\mathbf{z_h}} \circ h_{t-1})+W_{oc}(\frac 1 {1-d_c} {\mathbf{z_c}} \circ c_{t})+b_o)\\
    h_t & = \tanh(c_t)\circ o_t\\
  \end{split}
  \f$
   *
   * For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation
   * \param d Dropout rate \f$d_x\f$ for the input \f$x_t\f$
   * \param d_h Dropout rate \f$d_x\f$ for the output \f$h_t\f$
   * \param d_c Dropout rate \f$d_x\f$ for the cell \f$c_t\f$
   */
  void set_dropout(float d, float d_h, float d_c);
  /**
   * \brief Set all dropout rates to 0
   * \details This is equivalent to `set_dropout(0)` or `set_dropout(0,0,0)`
   *
   */
  void disable_dropout();
  /**
   * \brief Set dropout masks at the beginning of a sequence for a specific bathc size
   * \details If this function is not called on batched input, the same mask will be applied across
   * all batch elements. Use this to apply different masks to each batch element
   *
   * \param batch_size Batch size
   */
  void set_dropout_masks(unsigned batch_size = 1);
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

  // first index is layer, then ...
  // masks for Gal dropout
  std::vector<std::vector<Expression>> masks;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned input_dim = 0;
  unsigned hid = 0;

  float dropout_rate_h = 0.f, dropout_rate_c = 0.f;

private:
  DYNET_SERIALIZE_DECLARE()
  ComputationGraph  *_cg;

};


/**
 * \ingroup rnnbuilders
 * @brief VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections
 * @details This cell runs according to the following dynamics :
 *
 * \f$
 * \begin{split}
    i_t & =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)\\
    f_t & = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)\\
    o_t & = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)\\
    \tilde{c_t} & = \tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)\\
    c_t & = c_{t-1}\circ f_t + \tilde{c_t}\circ i_t\\
    h_t & = \tanh(c_t)\circ o_t\\
   \end{split}
 * \f$
 */
struct VanillaLSTMBuilder : public RNNBuilder {
  /**
   * @brief Default Constructor
   */
  VanillaLSTMBuilder();
  /**
   * \brief Constructor for the VanillaLSTMBuilder
   *
   * \param layers Number of layers
   * \param input_dim Dimention of the input \f$x_t\f$
   * \param hidden_dim Dimention of the hidden states \f$h_t\f$ and \f$c_t\f$
   * \param model Model holding the parameters
   * \param ln_lstm Whether to use layer normalization
   */
  explicit VanillaLSTMBuilder(unsigned layers,
                              unsigned input_dim,
                              unsigned hidden_dim,
                              Model& model,
                              bool ln_lstm = false);

  Expression back() const override { return (cur == -1 ? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const override { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const override {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    for (auto my_h : final_h()) ret.push_back(my_h);
    return ret;
  }
  unsigned num_h0_components() const override { return 2 * layers; }

  std::vector<Expression> get_h(RNNPointer i) const override { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const override {
    std::vector<Expression> ret = (i == -1 ? c0 : c[i]);
    for (auto my_h : get_h(i)) ret.push_back(my_h);
    return ret;
  }

  void copy(const RNNBuilder & params) override;

  void save_parameters_pretraining(const std::string& fname) const override;
  void load_parameters_pretraining(const std::string& fname) override;
  /**
   * \brief Set the dropout rates to a unique value
   * \details This has the same effect as `set_dropout(d,d_h)` except that all the dropout rates are set to the same value.
   * \param d Dropout rate to be applied on all of \f$x,h\f$
   */
  void set_dropout(float d);
  /**
   * \brief Set the dropout rates
   * \details The dropout implemented here is the variational dropout with tied weights introduced in [Gal, 2016](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks)
   * More specifically, dropout masks \f$\mathbf{z_x}\sim \mathrm{Bernoulli}(1-d_x)\f$,\f$\mathbf{z_h}\sim \mathrm{Bernoulli}(1-d_h)\f$ are sampled at the start of each sequence.
   * The dynamics of the cell are then modified to :
   *
   * \f$
   * \begin{split}
    i_t & =\sigma(W_{ix}(\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)\\
    f_t & = \sigma(W_{fx}(\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)\\
    o_t & = \sigma(W_{ox}(\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)\\
    \tilde{c_t} & = \tanh(W_{cx}(\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)\\
    c_t & = c_{t-1}\circ f_t + \tilde{c_t}\circ i_t\\
    h_t & = \tanh(c_t)\circ o_t\\
   \end{split}
   * \f$
   *
   * For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation
   * \param d Dropout rate \f$d_x\f$ for the input \f$x_t\f$
   * \param d_h Dropout rate \f$d_h\f$ for the output \f$h_t\f$
   */
  void set_dropout(float d, float d_r);
  /**
   * \brief Set all dropout rates to 0
   * \details This is equivalent to `set_dropout(0)` or `set_dropout(0,0,0)`
   *
   */
  void disable_dropout();
  /**
   * \brief Set dropout masks at the beginning of a sequence for a specific batch size
   * \details If this function is not called on batched input, the same mask will be applied across
   * all batch elements. Use this to apply different masks to each batch element
   *
   * \param batch_size Batch size
   */
  void set_dropout_masks(unsigned batch_size = 1);
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
  std::vector<std::vector<Parameter>> ln_params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;
  // first index is layer, then ...
  std::vector<std::vector<Expression>> ln_param_vars;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> masks;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned input_dim, hid;
  float dropout_rate_h;
  bool ln_lstm;



private:
  DYNET_SERIALIZE_DECLARE()
  ComputationGraph* _cg; // Pointer to current cg

};

} // namespace dynet


// Class version
DYNET_VERSION_DEFINE(dynet::LSTMBuilder, 1);
// Class version
DYNET_VERSION_DEFINE(dynet::VanillaLSTMBuilder, 1);


#endif
