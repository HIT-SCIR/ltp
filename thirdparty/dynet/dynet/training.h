/**
 * \file training.h
 * \defgroup optimizers
 * \brief Training procedures
 *
 * The various trainers are defined here.
 * All trainers are structures inheriting from the `Trainer` struct.
 *
 *
 */

#ifndef DYNET_TRAINING_H_
#define DYNET_TRAINING_H_

#include <vector>

#include <boost/serialization/export.hpp>

#include "dynet/model.h"
#include "dynet/shadow-params.h"
#include "dynet/io-macros.h"

#define DYNET_TRAINER_DEFINE_DEV_IMPL() \
  void update_params(real scale, real gscale, size_t idx) override; \
  void update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) override; \
  void update_lookup_params(real scale, real gscale, size_t idx) override; \
  template <class MyDevice> \
  void update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void update_rule(real scale, real gscale, const std::vector<Tensor*> & values) override;

namespace dynet {

/**
 * \ingroup optimizers
 *
 * \struct Trainer
 * \brief General trainer struct
 *
 */
struct Trainer {
  /**
   * \brief General constructor for a Trainer
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param edecay Learning rate decay
   */
  explicit Trainer(Model& m, real e0, real edecay = 0.0) :
    eta0(e0), eta(e0), eta_decay(edecay), epoch(), clipping_enabled(true), clip_threshold(5),
    clips(), updates(), clips_since_status(), updates_since_status(), sparse_updates_enabled(true), aux_allocated(false), model(&m) {}
  virtual ~Trainer();

  /**
   * \brief Update parameters
   * \details Update the parameters according to the appropriate update rule
   *
   * \param scale The scaling factor for the gradients
   */
  void update(real scale = 1.0);

  /**
   * \brief Update subset of parameters
   * \details Update some but not all of the parameters included in the model. This
   *        is the update_subset() function in the Python bindings. The
   *        parameters to be updated are specified by index, which can be found
   *        for Parameter and LookupParameter objects through the "index" variable
   *        (or the get_index() function in the Python bindings).
   *
   * \param updated_params The parameter indices to be updated
   * \param updated_lookup_params The lookup parameter indices to be updated
   * \param scale The scaling factor for the gradients
   */
  void update(const std::vector<unsigned> & updated_params, const std::vector<unsigned> & updated_lookup_params, real scale = 1.0);

  void update_epoch(real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  /**
   * \brief Clip gradient
   * \details If clipping is enabled and the gradient is too big, return the amount to
   *          scale the gradient by (otherwise 1)
   *
   *
   * \param scale The clipping limit
   * \return The appropriate scaling factor
   */
  float clip_gradients(real scale);

  // TODO: This is unprotected temporarily until there is a better solution
  //       for serializing the weight decay when saving models
  // Rescale all the parameters handled by this model
  void rescale_and_reset_weight_decay();

  // learning rates
  real eta0;
  real eta;
  real eta_decay;
  real epoch;

  // clipping
  bool clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;
  // the number of clips and status since the last print
  real clips_since_status;
  real updates_since_status;

  /**
   * \brief Whether to perform sparse updates
   * \details DyNet trainers support two types of updates for lookup parameters,
   *          sparse and dense. Sparse updates are the default. They have the
   *          potential to be faster, as they only touch the parameters that have
   *          non-zero gradients. However, they may not always be faster (particulary
   *          on GPU with mini-batch training), and are not precisely numerically
   *          correct for some update rules such as MomentumTrainer and AdamTrainer.
   *          Thus, if you set this variable to false, the trainer will perform dense
   *          updates and be precisely correct, and maybe faster sometimes.
   */
  bool sparse_updates_enabled;

  bool aux_allocated;

  void status() {
    std::cerr << "[epoch=" << epoch << " eta=" << eta << " clips=" << clips_since_status << " updates=" << updates_since_status << "] ";
    updates_since_status = clips_since_status = 0;
  }

  Model* model;  // parameters and gradients live here

protected:
  Trainer() {}
  virtual void alloc_impl() { }
  /**
   * \brief The actual rule to update the parameters
   *
   * \param scale Scale of the update (i.e. learning rate)
   * \param gscale Gradient scale based on clipping
   * \param values Values specific to the particular update rule being implemented
   */
  virtual void update_rule(real scale, real gscale, const std::vector<Tensor*> & values) = 0;
  /**
   * \brief Parameter update function
   *
   * \param scale Scale of the update (i.e. learning rate)
   * \param gscale Gradient scale based on clipping
   * \param idx Index of the parameter
   */
  virtual void update_params(real scale, real gscale, size_t idx) = 0;
  /**
   * \brief Sparse lookup parameter update function
   *
   * \param scale Scale of the update (i.e. learning rate)
   * \param gscale Gradient scale based on clipping
   * \param idx Index of the lookup parameter object
   * \param lidx Index of the specific entry within the lookup parameter object
   */
  virtual void update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) = 0;
  /**
   * \brief Dense lookup parameter update function
   *
   * \param scale Scale of the update (i.e. learning rate)
   * \param gscale Gradient scale based on clipping
   * \param idx Index of the lookup parameter object
   */
  virtual void update_lookup_params(real scale, real gscale, size_t idx) = 0;

private:
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief Stochastic gradient descent trainer
 * \details This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.
 * In the standard setting, the learning rate at epoch \f$t\f$ is \f$\eta_t=\frac{\eta_0}{1+\eta_{\mathrm{decay}}t}\f$
 *
 * Reference : [reference needed](ref.need.ed)
 *
 */
struct SimpleSGDTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param edecay Learning rate decay parameter.
   */
  explicit SimpleSGDTrainer(Model& m, real e0 = 0.1, real edecay = 0.0) : Trainer(m, e0, edecay) {}
protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
private:
  SimpleSGDTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief Cyclical learning rate SGD
 * \details This trainer performs stochastic gradient descent with a cyclical learning rate as proposed in [Smith, 2015](https://arxiv.org/abs/1506.01186).
 *
 * This uses a triangular function with optional exponential decay.
 *
 * More specifically, at each update, the learning rate \f$\eta\f$ is updated according to :
 *
 * \f$
 * \begin{split}
 * \text{cycle} &= \left\lfloor 1 + \frac{\texttt{it}}{2 \times\texttt{step_size}} \right\rfloor\\
 * x &= \left\vert \frac{\texttt{it}}{\texttt{step_size}} - 2 \times \text{cycle} + 1\right\vert\\
 * \eta &= \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \times \max(0, 1 - x) \times \gamma^{\texttt{it}}\\
 * \end{split}
 * \f$
 *
 *
 * Reference : [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
 *
 */
struct CyclicalSGDTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0_min Lower learning rate
   * \param e0_max Upper learning rate
   * \param step_size Period of the triangular function in number of iterations (__not__ epochs). According to the original paper, this should be set around (2-8) x (training iterations in epoch)
   * \param gamma Learning rate upper bound decay parameter
   * \param edecay Learning rate decay parameter. Ideally you shouldn't use this with cyclical learning rate since decay is already handled by \f$\gamma\f$
   */
  explicit CyclicalSGDTrainer(Model& m, float e0_min = 0.01, float e0_max = 0.1, float step_size = 2000, float gamma = 0.0, float edecay = 0.0) : Trainer(m, e0_min, edecay), e_min(e0_min), e_max(e0_max), step_size(step_size), gamma(gamma), it(0) {}
  void update(real scale = 1.0) { Trainer::update(scale);cyclic_update_eta();}
protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  void cyclic_update_eta() {
    float cycle = std::floor(1 + ((float) it)  / (2 * step_size));
    float x = std::abs( ((float) it) / step_size - 2 * cycle + 1);
    eta = e_min + ((1 - x) > 0 ? (e_max - e_min) * (1 - x) * (real)std::pow(gamma, it) : 0);
    it++;
  }
  float e_min;
  float e_max;
  float step_size;
  float gamma;
  unsigned it;
private:
  CyclicalSGDTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};


/**
 * \ingroup optimizers
 *
 * \brief Stochastic gradient descent with momentum
 * \details This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.
 * The modified gradient is \f$\theta_{t+1}=\mu\theta_{t}+\nabla_{t+1}\f$ where \f$\mu\f$ is the momentum.
 *
 * Reference : [reference needed](ref.need.ed)
 *
 */
struct MomentumSGDTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param mom Momentum
   * \param edecay Learning rate decay parameter
   */
  explicit MomentumSGDTrainer(Model& m, real e0 = 0.01, real mom = 0.9, real edecay = 0.0) :
    Trainer(m, e0, edecay), momentum(mom) {}

protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real momentum;

  // the following represent the current velocity
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
  //std::unordered_map<ParameterStorage*, Tensor> vp;
  //std::unordered_map<LookupParameterStorage*, std::unordered_map<unsigned, Tensor>> vl;
private:
  MomentumSGDTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief Adagrad optimizer
 * \details The adagrad algorithm assigns a different learning rate to each parameter according to the following formula :
 * \f$\delta_\theta^{(t)}=-\frac{\eta_0}{\epsilon+\sum_{i=0}^{t-1}(\nabla_\theta^{(i)})^2}\nabla_\theta^{(t)}\f$
 *
 * Reference : [Duchi et al., 2011](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
 *
 */
struct AdagradTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param eps Bias parameter \f$\epsilon\f$ in the adagrad formula
   * \param edecay Learning rate decay parameter
   */
  explicit AdagradTrainer(Model& m, real e0 = 0.1, real eps = 1e-20, real edecay = 0.0) :
    Trainer(m, e0, edecay), epsilon(eps) {}
protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
private:
  AdagradTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief AdaDelta optimizer
 * \details The AdaDelta optimizer is a variant of Adagrad where
 * \f$\frac{\eta_0}{\sqrt{\epsilon+\sum_{i=0}^{t-1}(\nabla_\theta^{(i)})^2}}\f$ is replaced by
 * \f$\frac{\sqrt{\epsilon+\sum_{i=0}^{t-1}\rho^{t-i-1}(1-\rho)(\delta_\theta^{(i)})^2}}{\sqrt{\epsilon+\sum_{i=0}^{t-1}(\nabla_\theta^{(i)})^2}}\f$,
 * hence eliminating the need for an initial learning rate.
 *
 * Reference : [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701v1)
 *
 */
struct AdadeltaTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param eps Bias parameter \f$\epsilon\f$ in the adagrad formula
   * \param rho Update parameter for the moving average of updates in the numerator
   * \param edecay Learning rate decay parameter
   */
  explicit AdadeltaTrainer(Model& m, real eps = 1e-6, real rho = 0.95, real edecay = 0.0) :
    Trainer(m, 1.0, edecay), epsilon(eps), rho(rho) {}
protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  real rho;
  std::vector<ShadowParameters> hg; // History of gradients
  std::vector<ShadowLookupParameters> hlg;
  std::vector<ShadowParameters> hd; // History of deltas
  std::vector<ShadowLookupParameters> hld;
private:
  AdadeltaTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief RMSProp optimizer
 * \details The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter \f$\rho\f$.
 *
 * Reference : [reference needed](ref.need.ed)
 *
 */
struct RMSPropTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param eps Bias parameter \f$\epsilon\f$ in the adagrad formula
   * \param rho Update parameter for the moving average (`rho = 0` is equivalent to using Adagrad)
   * \param edecay Learning rate decay parameter
   */
  explicit RMSPropTrainer(Model& m, real e0 = 0.001, real eps = 1e-08, real rho = 0.9, real edecay = 0.0) :
    Trainer(m, e0, edecay), epsilon(eps), rho(rho) {}
protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  real epsilon;
  real rho;
  std::vector<ShadowParameters> hmsg; // History of gradients
  std::vector<ShadowLookupParameters> hlmsg;
private:
  RMSPropTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup optimizers
 *
 * \brief Adam optimizer
 * \details The Adam optimizer is similar to RMSProp but uses unbiased estimates
 * of the first and second moments of the gradient
 *
 * Reference : [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980v8)
 *
 */
struct AdamTrainer : public Trainer {
  /**
   * \brief Constructor
   *
   * \param m Model to be trained
   * \param e0 Initial learning rate
   * \param beta_1 Moving average parameter for the mean
   * \param beta_2 Moving average parameter for the variance
   * \param eps Bias parameter \f$\epsilon\f$
   * \param edecay Learning rate decay parameter
   */
  explicit AdamTrainer(Model& m, float e0 = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8, real edecay = 0.0) :
    Trainer(m, e0, edecay), beta_1(beta_1), beta_2(beta_2), epsilon(eps) {}

protected:
  DYNET_TRAINER_DEFINE_DEV_IMPL()
  virtual void alloc_impl() override;

  float beta_1;
  float beta_2;
  float epsilon;
  std::vector<ShadowParameters> m; // History of gradients
  std::vector<ShadowLookupParameters> lm;
  std::vector<ShadowParameters> v; // History of deltas
  std::vector<ShadowLookupParameters> lv;
private:
  AdamTrainer() {}
  DYNET_SERIALIZE_DECLARE()
};

} // namespace dynet

BOOST_CLASS_EXPORT_KEY(dynet::SimpleSGDTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::CyclicalSGDTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::MomentumSGDTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::AdagradTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::AdadeltaTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::RMSPropTrainer)
BOOST_CLASS_EXPORT_KEY(dynet::AdamTrainer)

#endif
