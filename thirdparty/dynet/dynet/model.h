/**
 * \file model.h
 * \defgroup params params
 *
 */

#ifndef DYNET_PARAMS_H_
#define DYNET_PARAMS_H_

#include <vector>
#include <set>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <boost/serialization/export.hpp>

#include "dynet/io-macros.h"
#include "dynet/tensor.h"
#include "dynet/weight-decay.h"

namespace dynet {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParameterInit;

/**
 * \ingroup params
 * @brief This is the base class for ParameterStorage and LookupParameterStorage, the objects handling the actual parameters.
 * @details You can access the storage from any Parameter (resp. LookupParameter) class, use it only to do low level manipulations.
 *
 */
struct ParameterStorageBase {
  friend class Model;
  /**
   * @brief Scale the parameters
   *
   * @param a scale factor
   */
  virtual void scale_parameters(float a) = 0;
  /**
   * @brief Scale the gradient
   *
   * @param a scale factor
   */
  virtual void scale_gradient(float a) = 0;
  /**
   * @brief Set the parameters to 0
   */
  virtual void zero() = 0;
  /**
   * @brief Get the parameter squared l2 norm
   *
   * @param sqnorm Pointer to the float holding the result
   */
  virtual void squared_l2norm(float* sqnorm) const = 0;
  /**
   * @brief Get the squared l2 norm of the gradient w.r.t. these parameters
   *
   * @param sqnorm Pointer to the float holding the result
   */
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  /**
   * @brief Get the size (number of scalar parameters)
   * @return Number of scalar parameters
   */
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
  DYNET_SERIALIZE_COMMIT_EMPTY()
};

// represents parameters (e.g., a weight matrix) that will be optimized
/**
 * \ingroup params
 * \brief Storage class for Parameters
 */
struct ParameterStorage : public ParameterStorageBase {
  friend class Model;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  template <class MyDevice>
  void scale_gradient_dev(MyDevice & dev, float a);
  void scale_gradient(float a) override;
  void zero() override;
  template <class MyDevice>
  void squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void squared_l2norm(float* sqnorm) const override;
  template <class MyDevice>
  void g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  /**
   * @brief Copy from another ParameterStorage
   *
   * @param val ParameterStorage to copy from
   */
  void copy(const ParameterStorage & val);
  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, const Tensor& g);
  /**
   * @brief Add a tensor to the gradient
   * @details After this method gets called, g <- g + d
   *
   * @param g Tensor to add
   */
  void accumulate_grad(const Tensor& g);
  /**
   * @brief Clear the gradient (set it to 0)
   */
  void clear();
  /**
   * @brief Clip the values to the range [left, right]
   */
  void clip(float left, float right);
  
  Dim dim; /**< Dimensions of the parameter tensor*/
  Tensor values;/**< Values of the parameter */
  Tensor g;/**< Values of the gradient w.r.t. this parameter */

private:
  ParameterStorage() {}
  explicit ParameterStorage(const Dim& d, float minmax); // initialize with ~U(-minmax,+minmax)
  // or Glorot initialization if minmax = 0
  explicit ParameterStorage(const Dim& d, const ParameterInit & init); // initialize with custom initializer
  DYNET_SERIALIZE_DECLARE()
};

// represents a matrix/vector embedding of a discrete set
/**
 * \ingroup params
 * \brief Storage class for LookupParameters
 * 
 */
struct LookupParameterStorage : public ParameterStorageBase {
  friend class Model;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  template <class MyDevice>
  void scale_gradient_dev(MyDevice & dev, float a);
  void scale_gradient(float a) override;
  void zero() override;
  template <class MyDevice>
  void squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void squared_l2norm(float* sqnorm) const override;
  template <class MyDevice>
  void g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  template <class MyDevice>
  void initialize_dev(MyDevice & dev, unsigned index, const std::vector<float>& val);
  /**
   * @brief Initialize one particular lookup
   * 
   * @param index Index of the lookput to initialize
   * @param val Values
   */
  void initialize(unsigned index, const std::vector<float>& val);

  /**
   * @brief Copy from another LookupParameterStorage
   * 
   * @param val Other LookupParameterStorage to copy from
   */
  void copy(const LookupParameterStorage & val);

  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, const Tensor& g);
  /**
   * @brief Add a Tensor to the gradient of the whole lookup matrix
   * @details after this `grads<-grads + g`
   * 
   * @param g [description]
   */
  void accumulate_grad(const Tensor& g);

  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, unsigned index, const Tensor& g);
  /**
   * @brief Add a Tensor to the gradient of one of the lookups
   * @details after this `grads[index]<-grads[index] + g`
   * 
   * @param index [description]
   * @param g [description]
   */
  void accumulate_grad(unsigned index, const Tensor& g);
  template <class MyDevice>
  void accumulate_grads_dev(MyDevice & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  /**
   * @brief Add tensors to muliple lookups
   * @details After this method gets called, `grads[ids_host[i]] <- grads[ids_host[i]] + g[i*dim.size():(i+1)*dim.size()]`
   *
   * @param n size of `ids_host`
   * @param ids_host Indices of the gradients to update
   * @param ids_dev [To be documented] (only for GPU)
   * @param g Values
   */
  void accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  void clear();

  // Initialize each individual lookup from the overall tensors
  void initialize_lookups();

  // Tensors for all dimensions at once
  Dim all_dim; /**< Total dimension */
  Tensor all_values; /**< Values for all dimensions at once */
  Tensor all_grads; /**< Gradient values for all dimensions at once */
  // Tensors for each individual lookup
  Dim dim; /**< Dimension for one lookup */
  std::vector<Tensor> values; /**< List of values for each lookup */
  std::vector<Tensor> grads; /**< List of gradient values for each lookup */
  // gradients are sparse, so track which components are nonzero
  std::unordered_set<unsigned> non_zero_grads; /**< Gradients are sparse, so track which components are nonzero */
  bool all_updated; /** Whether all of the gradients have been updated. */
private:
  LookupParameterStorage() : all_updated(false) {}
  LookupParameterStorage(unsigned n, const Dim& d);
  LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit & init);
  DYNET_SERIALIZE_SPLIT_DECLARE()
};

class Model;
/**
 * \ingroup params
 * \brief Object representing a trainable parameter
 * \details This objects acts as a high level component linking the actual parameter values (ParameterStorage) and the Model. As long as you don't want to do low level hacks at the ParameterStorage level, this is what you will use.
 *
 */
struct Parameter {
  /**
   * @brief Default constructor
   */
  Parameter();
  /**
   * @brief Constructor
   * @details This is called by the model, you shouldn't need to use it
   *
   * @param mp Pointer to th model
   * @param index Id of the parameter
   */
  Parameter(Model* mp, unsigned long index);
  /**
   * @brief Get underlying ParameterStorage object
   * @return ParameterStorage holding the parameter values
   */
  ParameterStorage* get() const;

  /**
   * \brief Zero the parameters
   */
  void zero();

  Model* mp;/**< Pointer to the Model holding this parameter */
  unsigned long index;/**< Index of this parameter in its Model*/

  /**
   * \brief Shape of the parameter
   *
   * \return Shape as a `Dim` object
   */
  Dim dim() const { return get()->dim; }

  /**
   * \brief Values of the parameter
   *
   * \return Values as a `Tensor` object
   */
  Tensor* values() { return &(get()->values); }

  /**
   * @brief Set the parameter as updated
   *
   * @param b Update status
   */
  void set_updated(bool b);

  /**
   * @brief Scales the parameter (multiplies by `s`)
   *
   * @param s scale
   */
  void scale(float s){get()->scale_parameters(s);}


  /**
   * @brief Scales the gradient (multiplies by `s`)
   *
   * @param s scale
   */
  void scale_gradient(float s){get()->scale_gradient(s);}

  /**
   * @brief Check the update status
   * @return Update status
   */
  bool is_updated();
  /**
   * @brief Clip the values of the parameter to the range [left, right] (in place)
   */
  void clip_inplace(float left, float right);
private:
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup params
 * \brief Object representing a trainable lookup parameter
 *
 */
struct LookupParameter {
  LookupParameter();
  LookupParameter(Model* mp, unsigned long index);
  /**
   * @brief Get underlying LookupParameterStorage object
   * @return LookupParameterStorage holding the parameter values
   */
  LookupParameterStorage* get() const;
  /**
   * @brief Initialize one particular column
   *
   * @param index Index of the column to be initialized
   * @param val [description]
   */
  void initialize(unsigned index, const std::vector<float>& val) const;

  /**
   * \brief Zero the parameters
   */
  void zero();

  Model* mp;/**< Pointer to the Model holding this parameter */
  unsigned long index;/**< Index of this parameter in its Model*/

  /**
   * \brief Shape of the lookup parameter
   *
   * \return Shape as a `Dim` object
   */
  Dim dim() const { return get()->dim; }
  /**
   * \brief Values of the lookup parameter
   *
   * \return Values as a `Tensor` object
   */
  std::vector<Tensor>* values() { return &(get()->values); }

  /**
   * @brief Scales the parameter (multiplies by `s`)
   *
   * @param s scale
   */
  void scale(float s){get()->scale_parameters(s);}

  /**
   * @brief Scales the gradient (multiplies by `s`)
   *
   * @param s scale
   */
  void scale_gradient(float s){get()->scale_gradient(s);}

  /**
  * @brief Set the parameter as updated
  *
  * @param b Update status
  */
  void set_updated(bool b);
  /**
   * @brief Check the update status
   * @return Update status
   */
  bool is_updated();

private:
  DYNET_SERIALIZE_DECLARE()
};

/**
 * \ingroup params
 * \brief Initializers for parameters
 * \details Allows for custom parameter initialization
 */
struct ParameterInit {
  /**
   * \brief Default constructor
   */
  ParameterInit() {}
  virtual ~ParameterInit() {}
  /**
   * \brief Function called upon initialization
   * \details Whenever you inherit this struct to implement your own custom initializer, this is the function you want to overload to implement your logic.
   *
   * \param values The tensor to be initialized. You should modify it in-place. See dynet/model.cc for some examples
   */
  virtual void initialize_params(Tensor & values) const = 0;
};

/**
 * \ingroup params
 * \brief Initialize parameters with samples from a normal distribution
 */
struct ParameterInitNormal : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param m Mean of the gaussian distribution
   * \param v Variance of the gaussian distribution (reminder : the variance is the __square__ of the standard deviation)
   */
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float mean, var;
};

/**
 * \ingroup params
 * \brief Initialize parameters with samples from a uniform distribution
 *
 */
struct ParameterInitUniform : public ParameterInit {
  /**
   * \brief Constructor for uniform distribution centered on 0
   * \details [long description]Samples parameters from \f$mathcal U([-\mathrm{scale},+\mathrm{scale}]\f$
   * \param scale Scale of the distribution
   */
  ParameterInitUniform(float scale) :
    left(-scale), right(scale) { if (scale == 0.0f) throw std::domain_error("Scale of the uniform distribution cannot be 0 in ParameterInitUniform"); }
  /**
   * \brief Constructor for uniform distribution in a specific interval
   * \details [long description]
   *
   * \param l Lower bound of the interval
   * \param r Upper bound of the interval
   */
  ParameterInitUniform(float l, float r) : left(l), right(r) { if (l == r) throw std::domain_error("Empty interval in ParameterInitUniform"); }
  virtual void initialize_params(Tensor & values) const override;
private:
  float left, right;
};

/**
 * \ingroup params
 * \brief Initialize parameters with a constant value
 */
struct ParameterInitConst : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param c Constant value
   */
  ParameterInitConst(float c) : cnst(c) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};

/**
 * \ingroup params
 * \brief Initialize as the identity
 * \details This will raise an exception if used on non square matrices
 */
struct ParameterInitIdentity : public ParameterInit {
  /**
   * \brief Constructor
   */
  ParameterInitIdentity() {}
  virtual void initialize_params(Tensor & values) const override;
};

/**
 * \ingroup params
 * \brief Initialize with the methods described in [Glorot, 2010](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf?hc_location=ufi)
 * \details In order to preserve the variance of the forward and backward flow across layers, the parameters \f$\theta\f$ are initialized such that \f$\mathrm{Var}(\theta)=\frac 2 {n_1+n_2}\f$ where \f$n_1,n_2\f$ are the input and output dim.
 * Important note : The underlying distribution is uniform (not gaussian)
 *
 */
struct ParameterInitGlorot : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param is_lookup Boolean value identifying the parameter as a LookupParameter
   * \param gain Scaling parameter. In order for the Glorot initialization to be correct, you should ût this equal to \f$\frac 1 {f'(0)}\f$ where \f$f\f$ is your activation function
   */
  ParameterInitGlorot(bool is_lookup = false, float gain = 1.f) : lookup(is_lookup), gain(gain) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  bool lookup;
  float gain;
};
/**
 * \ingroup params
 * \brief Initializes according to [Saxe et al., 2014](https://arxiv.org/abs/1312.6120)
 * \details Initializes as a random orthogonal matrix (unimplemented for GPU)
 */
struct ParameterInitSaxe : public ParameterInit {
  /**
   * \brief Constructor
   */
  ParameterInitSaxe(float gain = 1.0) : gain(gain) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float gain;
};

/**
 * \ingroup params
 * \brief Initializes from a file
 * \details Useful for reusing weights, etc...
 *
 */
struct ParameterInitFromFile : public ParameterInit {
  /**
   * \brief Constructor
   * \param f File name (format should just be a list of values)
   */
  ParameterInitFromFile(std::string f) : filename(f) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::string filename;
};

/**
 * \ingroup params
 * \brief Initializes from a `std::vector` of floats
 */
struct ParameterInitFromVector : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param v Vector of values to be used
   */
  ParameterInitFromVector(std::vector<float> v) : vec(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::vector<float> vec;
};



// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
/**
 * \ingroup params
 * \brief This is a collection of parameters
 * \details if you need a matrix of parameters, or a lookup table - ask an instance of this class.
 * This knows how to serialize itself.
 * Parameters know how to track their gradients, but any extra information (like velocity) will live here
 */
class Model {
public:
  /**
   * \brief Constructor
   */
  Model();
  ~Model();
  template <class MyDevice>
  float gradient_l2_norm_dev(MyDevice & dev) const;
  /**
   * \brief Returns the l2 of your gradient
   * \details Use this to look for gradient vanishing/exploding
   * \return L2 norm of the gradient
   */
  float gradient_l2_norm() const;
  /**
   * \brief Sets all gradients to zero
   */
  void reset_gradient();
  // set scale to use custom initialization
  /**
   * \brief Add parameters to model and returns Parameter object
   * \details creates a ParameterStorage object holding a tensor of dimension `d` and returns a Parameter object (to be used as input in the computation graph). The coefficients are sampled according to the `scale` parameter
   *
   * \param d Shape of the parameter
   * \param scale If scale is non-zero, initializes according to \f$mathcal U([-\mathrm{scale},+\mathrm{scale}]\f$, otherwise uses Glorot initialization
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, float scale = 0.0f);
  /**
   * \brief Add parameters with custom initializer
   *
   * \param d Shape of the parameter
   * \param init Custom initializer
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, const ParameterInit & init);
  /**
   * \brief Add lookup parameter to model
   * \details Same as add_parameters. Initializes with Glorot
   *
   * \param n Number of lookup indices
   * \param d Dimension of each embedding
   *
   * \return LookupParameter object to be used in the computation graph
   */
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  /**
   * \brief Add lookup parameter with custom initializer
   *
   * \param n Number of lookup indices
   * \param d Dimension of each embedding
   * \param init Custom initializer
   * \return LookupParameter object to be used in the computation graph
   */
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init);
  //
  /**
   * \brief project weights so their L2 norm = radius
   * \details NOTE (Paul) : I am not sure this is doing anything currently. The argument doesn't seem to be used anywhere... If you need this raise an issue on github
   *
   * \param radius Target norm
   */
  void project_weights(float radius = 1.0f);
  /**
   * \brief Set the weight decay coefficient
   *
   * \param lambda Weight decay coefficient
   */
  void set_weight_decay_lambda(float lambda);

  //const std::vector<ParameterStorageBase*>& all_parameters_list() const { return all_params; }
  /**
   * \brief Returns list of pointers to ParameterSorages
   * \details You shouldn't need to use this
   * \return List of pointers to ParameterSorages
   */
  const std::vector<ParameterStorage*>& parameters_list() const { return params; }
  /**
   * \brief Returns list of pointers to LookupParameterSorages
   * \details You shouldn't need to use this
   * \return List of pointers to LookupParameterSorages
   */
  const std::vector<LookupParameterStorage*>& lookup_parameters_list() const { return lookup_params; }

  // indexes into params and lookup_params
  /**
   * \brief Returns list of indices of updated params
   *
   * \return list of indices of updated params
   */
  const std::vector<unsigned>& updated_parameters_list() const { return updated_params; }
  /**
   * \brief Returns list of indices of updated lookup params
   *
   * \return list of indices of updated lookup params
   */
  const std::vector<unsigned>& updated_lookup_parameters_list() const { return updated_lookup_params; }

  //
  //
  /**
   * \brief Returns the total number of tunable parameters (i. e. scalars) contained within this model.
   * \details That is to say, a 2x2 matrix counts as four parameters.
   * \return Number of parameters
   */
  size_t parameter_count() const;
  /**
   * \brief Returns total number of (scalar) parameters updated
   *
   * \return number of updated parameters
   */
  size_t updated_parameter_count() const;

  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \param status [description]
   */
  void set_updated_param(const Parameter *p, bool status);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \param status [description]
   */
  void set_updated_lookup_param(const LookupParameter *p, bool status);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \return [description]
   */
  bool is_updated_param(const Parameter *p);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \return [description]
   */
  bool is_updated_lookup_param(const LookupParameter *p);

  L2WeightDecay weight_decay;
private:
  DYNET_SERIALIZE_DECLARE()
  std::vector<ParameterStorageBase*> all_params;
  std::vector<ParameterStorage*> params;
  std::vector<LookupParameterStorage*> lookup_params;

  // these are a subset of the parameters that are used when model is updated.
  // kept as indices into params and lookup_params.
  std::vector<unsigned> updated_params;
  std::vector<unsigned> updated_lookup_params;

  mutable float* gradient_norm_scratch;
}; // class Model

void save_dynet_model(std::string filename, Model* model);
void load_dynet_model(std::string filename, Model* model);

} // namespace dynet

BOOST_CLASS_EXPORT_KEY(dynet::ParameterStorage)
BOOST_CLASS_EXPORT_KEY(dynet::LookupParameterStorage)

#endif
