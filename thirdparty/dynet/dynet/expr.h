/**
 * \file expr.h
 * \defgroup operations operations
 * \defgroup inputoperations inputoperations
 * \defgroup arithmeticoperations arithmeticoperations
 * \defgroup lossoperations lossoperations
 * \defgroup flowoperations flowoperations
 * \defgroup noiseoperations noiseoperations
 * \defgroup convolutionoperations convolutionoperations
 * \defgroup tensoroperations tensoroperations
 * \defgroup linalgoperations linalgoperations
 * \defgroup normoperations normoperations
 * \brief The various operations that you can use in building a DyNet graph
 *
 * \details TODO: **This documentation is incomplete. See expr.h for a full list of expressions.**
 */

#ifndef DYNET_EXPR_H
#define DYNET_EXPR_H

#include "dynet/dynet.h"
#include "dynet/nodes.h"
#include "dynet/nodes-contract.h"
#include <stdexcept>


namespace dynet {
namespace expr {
/**
 * \ingroup operations
 * \brief Expressions are the building block of a Dynet computation graph
 * \details [long description]
 */
struct Expression {
  ComputationGraph *pg;
  VariableIndex i;
  unsigned graph_id;

  Expression() : pg(nullptr), i(0), graph_id(0) { }
  const bool is_stale() const {return (get_number_of_active_graphs() != 1 || graph_id != get_current_graph_id());}
  /**
   * \brief Base expression constructor
   * \details Used when creating operations
   *
   * \param pg Pointer to the computation graph
   * \param i Variable index
   * \param name Name of the expression
   */
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i), graph_id(pg->get_id()) { }
  /**
   * \brief Get value of the expression
   * \details Throws a tuntime_error exception if no computation graph is available
   * \return Value of the expression as a tensor
   */
  const Tensor& value() const {
    if (this->is_stale()) {
      throw std::runtime_error("Attempt to use a stale expression.");
    }
    return pg->get_value(i);
  }
  /**
   * \brief Get gradient of the expression
   * \details Throws a tuntime_error exception if no computation graph is available
   * 
   * Make sure to call `backward` on a downstream expression before calling this.
   * 
   * If the expression is a constant expression (meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument `full=true` to `backward`
        
   * \return Value of the expression as a tensor
   */
  const Tensor& gradient() const {
    if (this->is_stale()) {
      throw std::runtime_error("Attempt to use a stale expression.");
    }
    return pg->get_gradient(i);
  }
  /**
   * \brief Get dimension of the expression
   * \details Throws a tuntime_error exception if no computation graph is available
   * \return Dimension of the expression
   */
  const Dim& dim() const {
    if (this->is_stale()) {
      throw std::runtime_error("Attempt to use a stale expression.");
    }
    return pg->get_dimension(i);
  }
};

namespace detail {
template <typename F, typename T>
Expression f(const T& xs) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  int i = 0;
  for (auto xi = xs.begin(); xi != xs.end(); ++xi) xis[i++] = xi->i;
  return Expression(pg, pg->add_function<F>(xis));
}
template <typename F, typename T, typename T1>
Expression f(const T& xs, const T1& arg1) {
  ComputationGraph *pg = xs.begin()->pg;
  std::vector<VariableIndex> xis(xs.size());
  int i = 0;
  for (auto xi = xs.begin(); xi != xs.end(); ++xi) xis[i++] = xi->i;
  return Expression(pg, pg->add_function<F>(xis, arg1));
}
}

////////////////////////////////////////////////
// Input operations                           //
////////////////////////////////////////////////

/**
 * \ingroup inputoperations
 * \brief Scalar input
 * \details Create an expression that represents the scalar value s
 *
 * \param g Computation graph
 * \param s Real number
 *
 * \return An expression representing s
 */
Expression input(ComputationGraph& g, real s);

/**
 * \ingroup inputoperations
 * \brief Modifiable scalar input
 * \details Create an expression that represents the scalar value *ps.
 *          If *ps is changed and the computation graph recalculated, the
 *          next forward pass will reflect the new value.
 *
 * \param g Computation graph
 * \param ps Real number pointer
 *
 * \return An expression representing *ps
 */
Expression input(ComputationGraph& g, const real *ps);

/**
 * \ingroup inputoperations
 * \brief Vector/matrix/tensor input
 * \details Create an expression that represents a vector, matrix, or tensor
 *          input. The dimensions of the input are defined by ``d``. So for example
 *          > ``input(g,{50},data)``: will result in a 50-length vector
 *          > ``input(g,{50,30},data)``: will result in a 50x30 matrix
 *          and so on, for an arbitrary number of dimensions.
 *          This function can also be used to import minibatched inputs. For example,
 *          if we have 10 examples in a minibatch, each with size 50x30, then we call
 *          > ``input(g,Dim({50,30},10),data)``
 *          The data vector "data" will contain the values used to fill the input, in
 *          column-major format. The length must add to the product of all dimensions in
 *          d.
 *
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param data A vector of data points
 *
 * \return An expression representing data
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>& data);

/**
 * \ingroup inputoperations
 * \brief Updatable vector/matrix/tensor input
 * \details Similarly to input that takes a vector reference, input a vector, matrix,
 *          or tensor input. Because we pass the pointer, the data can be updated.
 *
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param pdata A pointer to an (updatable) vector of data points
 *
 * \return An expression representing *pdata
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);

/**
 * \ingroup inputoperations
 * \brief Sparse vector input
 * \details This operation takes input as a sparse matrix of index/value pairs. It is
 *          exactly the same as the standard input via vector reference, but sets all
 *          non-specified values to "defdata" and resets all others to the appropriate
 *          input values.
 *
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param ids The indexes of the data points to update
 * \param data The data points corresponding to each index
 * \param defdata The default data with which to set the unspecified data points
 *
 * \return An expression representing data
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

/**
 * \ingroup inputoperations
 * \brief Load parameter
 * \details Load parameters into the computation graph.
 *
 * \param g Computation graph
 * \param p Parameter object to load
 *
 * \return An expression representing p
 */
Expression parameter(ComputationGraph& g, Parameter p);

/**
 * \ingroup inputoperations
 * \brief Load lookup parameter
 * \details Load a full tensor of lookup parameters into the computation graph.
 *          Normally lookup parameters are accessed by using the lookup() function
 *          to grab a single element. However, in some cases we'll want to access
 *          all of the parameters in the entire set of lookup parameters for some
 *          reason. In this case you can use this function. In this case, the
 *          first dimensions in the returned tensor will be equivalent to the
 *          dimensions that we would get if we get calling the lookup() function,
 *          and the size of the final dimension will be equal to the size of the
 *          vocabulary.
 *
 * \param g Computation graph
 * \param lp LookupParameter object to load
 *
 * \return An expression representing lp
 */
Expression parameter(ComputationGraph& g, LookupParameter lp);

/**
 * \ingroup inputoperations
 * \brief Load constant parameters
 * \details Load parameters into the computation graph, but prevent them from being
 *          updated when performing parameter update.
 *
 * \param g Computation graph
 * \param p Parameter object to load
 *
 * \return An expression representing the constant p
 */
Expression const_parameter(ComputationGraph& g, Parameter p);

/**
 * \ingroup inputoperations
 * \brief Load constant lookup parameters
 * \details Load lookup parameters into the computation graph, but prevent them from being
 *          updated when performing parameter update.
 *
 * \param g Computation graph
 * \param lp LookupParameter object to load
 *
 * \return An expression representing the constant lp
 */
Expression const_parameter(ComputationGraph& g, LookupParameter lp);

/**
 * \ingroup inputoperations
 * \brief Look up parameter
 * \details Look up parameters according to an index, and load them into the
 *          computation graph.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param index Index of the parameters within p
 *
 * \return An expression representing p[index]
 */
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index);

/**
 * \ingroup inputoperations
 * \brief Look up parameters with modifiable index
 * \details Look up parameters according to the *pindex, and load them into the
 *          computation graph. When *pindex changes, on the next computation of
 *          forward() the values will change.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindex Pointer index of the parameters within p
 *
 * \return An expression representing p[*pindex]
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);

/**
 * \ingroup inputoperations
 * \brief Look up parameter
 * \details Look up parameters according to an index, and load them into the
 *          computation graph. Do not perform gradient update on the parameters.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param index Index of the parameters within p
 *
 * \return A constant expression representing p[index]
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index);

/**
 * \ingroup inputoperations
 * \brief Constant lookup parameters with modifiable index
 * \details Look up parameters according to the *pindex, and load them into the
 *          computation graph. When *pindex changes, on the next computation of
 *          forward() the values will change. However, gradient updates will not be
            performend.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindex Pointer index of the parameters within p
 *
 * \return A constant expression representing p[*pindex]
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);

// Batched versions of lookup and const_lookup

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details The mini-batched version of lookup. The resulting expression will be
 *          a mini-batch of parameters, where the "i"th element of the batch corresponds
 *          to the parameters at the position specified by the "i"th element of
 *          "indices"
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param indices Index of the parameters at each position in the batch
 *
 * \return An expression with the "i"th batch element representing p[indices[i]]
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details The mini-batched version of lookup with modifiable parameter indices.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindices Pointer to lookup indices
 *
 * \return An expression with the "i"th batch element representing p[*pindices[i]]
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details Mini-batched lookup that will not update the parameters.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param indices Lookup indices
 *
 * \return A constant expression with the "i"th batch element representing p[indices[i]]
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details Mini-batched lookup that will not update the parameters, with modifiable
 *          indices.
 *
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindices Lookup index pointers.
 *
 * \return A constant expression with the "i"th batch element representing
 *         p[*pindices[i]]
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

/**
 * \ingroup inputoperations
 * \brief Create an input full of zeros
 * \details Create an input full of zeros, sized according to dimensions d.
 *
 * \param g Computation graph
 * \param d The dimensions of the input
 *
 * \return A "d" dimensioned zero vector
 */
Expression zeroes(ComputationGraph& g, const Dim& d);

/**
 * \ingroup inputoperations
 * \brief Create a random normal vector
 * \details Create a vector distributed according to normal distribution with mean
 *          0, variance 1.
 *
 * \param g Computation graph
 * \param d The dimensions of the input
 *
 * \return A "d" dimensioned normally distributed vector
 */
Expression random_normal(ComputationGraph& g, const Dim& d);

/**
 * \ingroup inputoperations
 * \brief Create a random bernoulli vector
 * \details Create a vector distributed according to bernoulli distribution with parameter p.
 *
 * \param g Computation graph
 * \param d The dimensions of the input
 * \param p The bernoulli p parameter
 * \param scale A scaling factor for the output ("active" elements will receive this value)
 *
 * \return A "d" dimensioned bernoulli distributed vector
 */
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale = 1.0f);

/**
 * \ingroup inputoperations
 * \brief Create a random uniform vector
 * \details Create a vector distributed according to uniform distribution with boundaries left and right.
 *
 * \param g Computation graph
 * \param d The dimensions of the input
 * \param left The left boundary
 * \param right The right boundary
 *
 * \return A "d" dimensioned uniform distributed vector
 */
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right);

/**
 * \ingroup inputoperations
 * \brief Create a random Gumbel sampled vector
 * \details Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported.
 *
 * \param g Computation graph
 * \param d The dimensions of the input
 * \param mu The mu parameter
 * \param beta The beta parameter
 *
 * \return A "d" dimensioned Gumbel distributed vector
 */
Expression random_gumbel(ComputationGraph& g, const Dim& d, real mu = 0.0, real beta = 1.0);

////////////////////////////////////////////////
// Arithmetic operations                      //
////////////////////////////////////////////////

/**
 * \ingroup arithmeticoperations
 * \brief Negation
 * \details Negate the passed argument.
 *
 * \param x An input expression
 *
 * \return The negation of x
 */
Expression operator-(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Expression addition
 * \details Add two expressions of the same dimensions.
 *
 * \param x The first input
 * \param y The second input
 *
 * \return The sum of x and y
 */
Expression operator+(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar addition
 * \details Add a scalar to an expression
 *
 * \param x The expression
 * \param y The scalar
 *
 * \return An expression equal to x, with every component increased by y
 */
Expression operator+(const Expression& x, real y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar addition
 * \details Add a scalar to an expression
 *
 * \param x The scalar
 * \param y The expression
 *
 * \return An expression equal to y, with every component increased by x
 */
Expression operator+(real x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Expression subtraction
 * \details Subtract one expression from another.
 *
 * \param x The expression from which to subtract
 * \param y The expression to subtract
 *
 * \return An expression where the ith element is x_i minus y_i
 */
Expression operator-(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar subtraction
 * \details Subtract an expression from a scalar
 *
 * \param x The scalar from which to subtract
 * \param y The expression to subtract
 *
 * \return An expression where the ith element is x_i minus y
 */
Expression operator-(real x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar subtraction
 * \details Subtract a scalar from an expression
 *
 * \param x The expression from which to subtract
 * \param y The scalar to subtract
 *
 * \return An expression where the ith element is x_i minus y
 */
Expression operator-(const Expression& x, real y);


/**
 * \ingroup arithmeticoperations
 * \brief Matrix multiplication
 * \details Multiply two matrices together. Like standard matrix multiplication, the
 *          second dimension of x and the first dimension of y must match.
 *
 * \param x The left-hand matrix
 * \param y The right-hand matrix
 *
 * \return An expression x times y
 */
Expression operator*(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar multiplication
 * \details Multiply an expression component-wise by a scalar.
 *
 * \param x The matrix
 * \param y The scalar
 *
 * \return An expression where the ith element is x_i times y
 */
Expression operator*(const Expression& x, float y);

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar multiplication
 * \details Multiply an expression component-wise by a scalar.
 *
 * \param x The scalar
 * \param y The matrix
 *
 * \return An expression where the ith element is x_i times y
 */
inline Expression operator*(float y, const Expression& x) { return x * y; }

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar division
 * \details Divide an expression component-wise by a scalar.
 *
 * \param x The matrix
 * \param y The scalar
 *
 * \return An expression where the ith element is x_i divided by y
 */
inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }

/**
 * \ingroup arithmeticoperations
 * \brief Affine transform
 * \details This performs an affine transform over an arbitrary (odd) number of expressions
 *          held in the input initializer list xs.
 *          The first expression is the "bias," which is added to the expression as-is.
 *          The remaining expressions are multiplied together in pairs, then added.
 *          A very common usage case is the calculation of the score for a neural network
 *          layer (e.g. b + Wz) where b is the bias, W is the weight matrix, and z is the
 *          input. In this case xs[0] = b, xs[1] = W, and xs[2] = z.
 *
 * \param xs An initializer list containing an odd number of expressions
 *
 * \return An expression equal to: xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...
 */
inline Expression affine_transform(const std::initializer_list<Expression>& xs) { return detail::f<AffineTransform>(xs); }
template <typename T>
inline Expression affine_transform(const T& xs) { return detail::f<AffineTransform>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Sum
 * \details This performs an elementwise sum over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to xs[0][i] + xs[1][i] + ...
 */
inline Expression sum(const std::initializer_list<Expression>& xs) { return detail::f<Sum>(xs); }
template <typename T>
inline Expression sum(const T& xs) { return detail::f<Sum>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Sum all elements
 * \details Sum all the elements in an expression.
 *
 * \param x The input expression
 *
 * \return The sum of all of its elements
 */
Expression sum_elems(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Compute moment over all elements
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 * \param r Order of the moment
 *
 * \return A scalar expression (with a potential batch dimension)
 */
Expression moment_elems(const Expression& x, unsigned r);

/**
 * \ingroup arithmeticoperations
 * \brief Compute mean over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
Expression mean_elems(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Compute Standard deviation over all elements
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ over all the elements in each batch of the expression
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
Expression std_elems(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Average
 * \details This performs an elementwise average over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to (xs[0][i] + xs[1][i] + ...)/|xs|
 */
inline Expression average(const std::initializer_list<Expression>& xs) { return detail::f<Average>(xs); }
template <typename T>
inline Expression average(const T& xs) { return detail::f<Average>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Square root
 * \details Elementwise square root.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\sqrt(x_i)\f$
 */
Expression sqrt(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Absolute value
 * \details Elementwise absolute value.
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to \f$\vert x_i\vert\f$
 */
Expression abs(const Expression& x);


/**
 * \ingroup arithmeticoperations
 * \brief Gaussian error function
 * \details Elementwise calculation of the Gaussian error function
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to erf(x_i)
 */
Expression erf(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic tangent
 * \details Elementwise calculation of the hyperbolic tangent
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to tanh(x_i)
 */
Expression tanh(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Natural exponent
 * \details Calculate elementwise y_i = e^{x_i}
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to e^{x_i}
 */
Expression exp(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Square
 * \details Calculate elementwise y_i = x_i^2
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^2
 */
Expression square(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Cube
 * \details Calculate elementwise y_i = x_i^3
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i^3
 */
Expression cube(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Log gamma
 * \details Calculate elementwise y_i = ln(gamma(x_i))
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(gamma(x_i))
 */
Expression lgamma(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Logarithm
 * \details Calculate the elementwise natural logarithm y_i = ln(x_i)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to ln(x_i)
 */
Expression log(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Logistic sigmoid function
 * \details Calculate elementwise y_i = 1/(1+e^{-x_i})
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to y_i = 1/(1+e^{-x_i})
 */
Expression logistic(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Rectifier
 * \details Calculate elementwise the recitifer (ReLU) function y_i = max(x_i,0)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to max(x_i,0)
 */
Expression rectify(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Soft Sign
 * \details Calculate elementwise the softsign function y_i = x_i/(1+|x_i|)
 *
 * \param x The input expression
 *
 * \return An expression where the ith element is equal to x_i/(1+|x_i|)
 */
Expression softsign(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Power function
 * \details Calculate an output where the ith element is equal to x_i^y_i
 *
 * \param x The input expression
 * \param y The exponent expression
 *
 * \return An expression where the ith element is equal to x_i^y_i
 */
Expression pow(const Expression& x, const Expression& y);
#undef min
#undef max
/**
 * \ingroup arithmeticoperations
 * \brief Minimum
 * \details Calculate an output where the ith element is min(x_i,y_i)
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to min(x_i,y_i)
 */
Expression min(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Maximum
 * \details Calculate an output where the ith element is max(x_i,y_i)
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to max(x_i,y_i)
 */
Expression max(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Max
 * \details This performs an elementwise max over all the expressions in xs
 *
 * \param xs An initializer list containing expressions
 *
 * \return An expression where the ith element is equal to max(xs[0][i], xs[1][i], ...)
 */
inline Expression max(const std::initializer_list<Expression>& xs) { return detail::f<Max>(xs); }
template <typename T>
inline Expression max(const T& xs) { return detail::f<Max>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Dot Product
 * \details Calculate the dot product sum_i x_i*y_i
 *
 * \param x The input expression
 * \param y The input expression
 *
 * \return An expression equal to the dot product
 */
Expression dot_product(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Componentwise multiply
 * \details Do a componentwise multiply where each value is equal to x_i*y_i.
 *          This function used to be called cwise_multiply.
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to x_i*y_i
 */
Expression cmult(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Componentwise multiply
 * \details Do a componentwise multiply where each value is equal to x_i/y_i
 *
 * \param x The first input expression
 * \param y The second input expression
 *
 * \return An expression where the ith element is equal to x_i/y_i
 */
Expression cdiv(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Columnwise addition
 * \details Add vector "bias" to each column of matrix "x"
 *
 * \param x An MxN matrix
 * \param bias A length M vector
 *
 * \return An expression where bias is added to each column of x
 */
Expression colwise_add(const Expression& x, const Expression& bias);

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

/**
 * \ingroup lossoperations
 * \brief Softmax
 * \details The softmax function normalizes each column to ensure that all
 *          values are between 0 and 1 and add to one by applying the
 *          e^{x[i]}/{sum_j e^{x[j]}}.
 *
 * \param x A vector or matrix
 *
 * \return A vector or matrix after calculating the softmax
 */
Expression softmax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Log softmax
 * \details The log softmax function normalizes each column to ensure that all
 *          values are between 0 and 1 and add to one by applying the
 *          e^{x[i]}/{sum_j e^{x[j]}}, then takes the log
 *
 * \param x A vector or matrix
 *
 * \return A vector or matrix after calculating the log softmax
 */
Expression log_softmax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Restricted log softmax
 * \details The log softmax function calculated over only a subset of the vector elements. The
 *          elements to be included are set by the ``restriction`` variable. All elements not
 *          included in ``restriction`` are set to negative infinity.
 *
 * \param x A vector over which to calculate the softmax
 * \param restriction The elements over which to calculate the softmax
 *
 * \return A vector with the log softmax over the specified elements
 */
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);

/**
 * \ingroup lossoperations
 * \brief Log, sum, exp
 * \details The elementwise "logsumexp" function that calculates
 *   \f$ln(\sum_i e^{xs_i})\f$, used in adding probabilities in the log domain.
 *
 * \param xs Expressions with respect to which to calculate the logsumexp.
 *
 * \return The result.
 */
inline Expression logsumexp(const std::initializer_list<Expression>& xs) { return detail::f<LogSumExp>(xs); }
template <typename T>
inline Expression logsumexp(const T& xs) { return detail::f<LogSumExp>(xs); }

/**
 * \ingroup lossoperations
 * \brief Negative softmax log likelihood
 * \details This function takes in a vector of scores ``x``, and performs a log softmax, takes
 *          the negative, and selects the likelihood corresponding to the element ``v``. This is
 *          perhaps the most standard loss function for training neural networks to predict
 *          one out of a set of elements.
 *
 * \param x A vector of scores
 * \param v The element with which to calculate the loss
 *
 * \return The negative log likelihood of element ``v`` after taking the softmax
 */
Expression pickneglogsoftmax(const Expression& x, unsigned v);

/**
 * \ingroup lossoperations
 * \brief Modifiable negative softmax log likelihood
 * \details This function calculates the negative log likelihood after the softmax with
 *          respect to index ``*pv``. This computes the same value as the previous function
 *          that passes the index ``v`` by value, but instead passes by pointer so the value
 *          ``*pv`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 *
 * \param x A vector of scores
 * \param pv A pointer to the index of the correct element
 *
 * \return The negative log likelihood of element ``*pv`` after taking the softmax
 */
Expression pickneglogsoftmax(const Expression& x, const unsigned * pv);

/**
 * \ingroup lossoperations
 * \brief Batched negative softmax log likelihood
 * \details This function is similar to standard pickneglogsoftmax, but calculates loss with
 *          respect to multiple batch elements. The input will be a mini-batch of score vectors
 *          where the number of batch elements is equal to the number of indices in ``v``.
 *
 * \param x An expression with vectors of scores over N batch elements
 * \param v A size-N vector indicating the index with respect to all the batch elements
 *
 * \return The negative log likelihoods over all the batch elements
 */
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> & v);

/**
 * \ingroup lossoperations
 * \brief Modifiable batched negative softmax log likelihood
 * \details This function is a combination of modifiable pickneglogsoftmax and batched
 *          pickneglogsoftmax: ``pv`` can be modified without re-creating the computation graph.
 *
 * \param x An expression with vectors of scores over N batch elements
 * \param pv A pointer to the indexes
 *
 * \return The negative log likelihoods over all the batch elements
 */
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> * pv);

/**
 * \ingroup lossoperations
 * \brief Hinge loss
 * \details This expression calculates the hinge loss, formally expressed as:
 *          \f$ \text{hinge}(x,index,m) = \sum_{i \ne index} \max(0, m-x[index]+x[i]). \f$
 *
 * \param x A vector of scores
 * \param index The index of the correct candidate
 * \param m The margin
 *
 * \return The hinge loss of candidate ``index`` with respect to margin ``m``
 */
Expression hinge(const Expression& x, unsigned index, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Modifiable hinge loss
 * \details This function calculates the hinge loss with
 *          with respect to index ``*pindex``. This computes the same value as the previous function
 *          that passes the index ``index`` by value, but instead passes by pointer so the value
 *          ``*pindex`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 *
 * \param x A vector of scores
 * \param pindex A pointer to the index of the correct candidate
 * \param m The margin
 *
 * \return The hinge loss of candidate ``*pindex`` with respect to margin ``m``
 */
Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Batched hinge loss
 * \details The same as hinge loss, but for the case where ``x`` is a mini-batched tensor
 *          with ``indices.size()`` batch elements, and ``indices`` is a vector indicating
 *          the index of each of the correct elements for these elements.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param indices The indices of the correct candidates for each batch element
 * \param m The margin
 *
 * \return The hinge loss of each mini-batch
 */
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Batched modifiable hinge loss
 * \details A combination of the previous batched and modifiable hinge loss functions, where
 *          vector ``*pindices`` can be modified.
 *
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param pindices Pointer to the indices of the correct candidates for each batch element
 * \param m The margin
 *
 * \return The hinge loss of each mini-batch
 */
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Sparsemax
 * \details The sparsemax function (Martins et al. 2016), which is similar to softmax,
 *          but induces sparse solutions where most of the vector elements are zero.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 *
 * \return The sparsemax of the scores
 */
Expression sparsemax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Sparsemax loss
 * \details The sparsemax loss function (Martins et al. 2016), which is similar to
 *          softmax loss, but induces sparse solutions where most of the vector
 *          elements are zero. It has a gradient similar to the sparsemax function
 *          and thus is useful for optimizing when the sparsemax will be used at
 *          test time.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 * \param target_support The target correct labels.
 *
 * \return The sparsemax loss of the labels
 */
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>& target_support);

/**
 * \ingroup lossoperations
 * \brief Modifiable sparsemax loss
 * \details Similar to the sparsemax loss, but with ptarget_support being a pointer
 *          to a vector, allowing it to be modified without re-creating the compuation
 *          graph.
 *          **Note:** This function is not yet implemented on GPU.
 *
 * \param x A vector of scores
 * \param ptarget_support A pointer to the target correct labels.
 *
 * \return The sparsemax loss of the labels
 */
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>* ptarget_support);

/**
 * \ingroup lossoperations
 * \brief Squared norm
 * \details The squared norm of the values of x: \f$\sum_i x_i^2\f$.
 *
 * \param x A vector of values
 *
 * \return The squared norm
 */
Expression squared_norm(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Squared distance
 * \details The squared distance between values of ``x`` and ``y``: \f$\sum_i (x_i-y_i)^2\f$.
 *
 * \param x A vector of values
 * \param y Another vector of values
 *
 * \return The squared distance
 */
Expression squared_distance(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief L1 distance
 * \details The L1 distance between values of ``x`` and ``y``: \f$\sum_i |x_i-y_i|\f$.
 *
 * \param x A vector of values
 * \param y Another vector of values
 *
 * \return The squared distance
 */
Expression l1_distance(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief Huber distance
 * \details The huber distance between values of ``x`` and ``y`` parameterized
 *    by ``c,`` \f$\sum_i L_c(x_i, y_i)\f$ where:
 *
 *    \f$
 *      L_c(x, y) = \begin{cases}{lr}
 *        \frac{1}{2}(y - x)^2                   & \textrm{for } |y - f(x)| \le c, \\
 *        c\, |y - f(x)| - \frac{1}{2}c^2 & \textrm{otherwise.}
 *      \end{cases}
 *    \f$
 *
 * \param x A vector of values
 * \param y Another vector of values
 * \param c The parameter of the huber distance parameterizing the cuttoff
 *
 * \return The huber distance
 */
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);

/**
 * \ingroup lossoperations
 * \brief Binary log loss
 * \details The log loss of a binary decision according to the sigmoid
 *          sigmoid function \f$- \sum_i (y_i * ln(x_i) + (1-y_i) * ln(1-x_i)) \f$
 *
 * \param x A vector of values
 * \param y A vector of true answers
 *
 * \return The log loss of the sigmoid function
 */
Expression binary_log_loss(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief Pairwise rank loss
 * \details A margin-based loss, where every margin violation for each pair of
 *          values is penalized: \f$\sum_i max(x_i-y_i+m, 0)\f$
 *
 * \param x A vector of values
 * \param y A vector of true answers
 * \param m The margin
 *
 * \return The pairwise rank loss
 */
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Poisson loss
 * \details The negative log probability of ``y`` according to a Poisson
 *          distribution with parameter ``x``. Useful in Poisson regression
 *          where, we try to predict the parameters of a Possion distribution
 *          to maximize the probability of data ``y``.
 *
 * \param x The parameter of the Poisson distribution.
 * \param y The target value
 *
 * \return The Poisson loss
 */
Expression poisson_loss(const Expression& x, unsigned y);
/**
 * \ingroup lossoperations
 * \brief Modifiable Poisson loss
 * \details Similar to Poisson loss, but with the target value passed by
 *          pointer so that it can be modified without re-constructing the
 *          computation graph.
 *
 * \param x The parameter of the Poisson distribution.
 * \param py A pointer to the target value
 *
 * \return The Poisson loss
 */
Expression poisson_loss(const Expression& x, const unsigned* py);

////////////////////////////////////////////////
// Flow operations                            //
////////////////////////////////////////////////

/**
 * \ingroup flowoperations
 * \brief Prevent backprop
 * \details This node has no effect on the forward pass, but prevents gradients from
 *          flowing backward during the backward pass. This is useful when there's
 *          a subgraph for which you don't want loss passed back to the parameters.
 *
 * \param x The input expression
 *
 * \return The new expression
 */
Expression nobackprop(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Negative backprop
 * \details This node has no effect on the forward pass, but takes negative on backprop process.
 *          This operation is widely used in adversarial networks.
 *
 * \param x The input expression
 *
 * \return An output expression containing the same as input (only effects on backprop process)
 */
Expression flip_gradient(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Reshape to another size
 * \details This node reshapes a tensor to another size, without changing the
 *          underlying layout of the data. The layout of the data in DyNet is
 *          column-major, so if we have a 3x4 matrix
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
 *        x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
 *        x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *          and transform it into a 2x6 matrix, it will be rearranged as:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{3,1} & x_{2,2} & x_{1,3} & x_{3,3} & x_{2,4} \\
 *        x_{2,1} & x_{1,2} & x_{3,2} & x_{2,3} & x_{1,4} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *         **Note:** This is O(1) for forward, and O(n) for backward.
 *
 * \param x The input expression
 * \param d The new dimensions
 *
 * \return The reshaped expression
 */
Expression reshape(const Expression& x, const Dim& d);

/**
 * \ingroup flowoperations
 * \brief Transpose a matrix
 * \details Transpose a matrix or tensor, or if dims is specified shuffle the
 *          dimensions arbitrarily.
 *          **Note:** This is O(1) if either the row or column dimension is 1,
 *          and O(n) otherwise.
 *
 * \param x The input expression
 * \param dims The dimensions to swap. The ith dimension of the output will be equal
 *          to the dims[i] dimension of the input. dims must have the same number
 *          of dimensions as x.
 *
 * \return The transposed/shuffled expression
 */
Expression transpose(const Expression& x, const std::vector<unsigned> & dims = {1,0});

/**
 * \ingroup flowoperations
 * \brief Select rows
 * \details Select a subset of rows of a matrix.
 *
 * \param x The input expression
 * \param rows The rows to extract
 *
 * \return An expression containing the selected rows
 */
Expression select_rows(const Expression& x, const std::vector<unsigned>& rows);

/**
 * \ingroup flowoperations
 * \brief Modifiable select rows
 * \details Select a subset of rows of a matrix, where the elements of prows
 *          can be modified without re-creating the computation graph.
 *
 * \param x The input expression
 * \param prows The rows to extract
 *
 * \return An expression containing the selected rows
 */
Expression select_rows(const Expression& x, const std::vector<unsigned>* prows);

/**
 * \ingroup flowoperations
 * \brief Select columns
 * \details Select a subset of columns of a matrix. select_cols is more
 *          efficient than select_rows since DyNet uses column-major order.
 *
 * \param x The input expression
 * \param columns The columns to extract
 *
 * \return An expression containing the selected columns
 */
Expression select_cols(const Expression& x, const std::vector<unsigned>& cols);

/**
 * \ingroup flowoperations
 * \brief Modifiable select columns
 * \details Select a subset of columns of a matrix, where the elements of pcols
 *          can be modified without re-creating the computation graph.
 *
 * \param x The input expression
 * \param pcolumns The columns to extract
 *
 * \return An expression containing the selected columns
 */
Expression select_cols(const Expression& x, const std::vector<unsigned>* pcols);

/**
 * \ingroup flowoperations
 * \brief Sum over minibatches
 * \details Sum an expression that consists of multiple minibatches into one of
 *          equal dimension but with only a single minibatch. This is useful
 *          for summing loss functions at the end of minibatch training.
 *
 * \param x The input mini-batched expression
 *
 * \return An expression with a single batch
 */
Expression sum_batches(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Compute moment over minibatches
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ along the batch dimension 
 *
 * \param x The input mini-batched expression
 * \param r Order of the moment
 *
 * \return An expression with a single batch
 */
Expression moment_batches(const Expression& x, unsigned r);


/**
 * \ingroup flowoperations
 * \brief Compute mean over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension 
 *
 * \param x The input mini-batched expression
 *
 * \return An expression with a single batch
 */
Expression mean_batches(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Compute standard deviation over minibatches
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ along the batch dimension 
 *
 * \param x The input mini-batched expression
 *
 * \return A scalar expression (with a potential batch dimension)
 */
Expression std_batches(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Compute standard deviation along an arbitrary dimension
 * \details Computes \f$\frac 1 n\sum_{i=1}^n(x_i -\mu)^2\f$ where \f$\mu=\frac 1 n\sum_{i=1}^nx_i\f$ along an arbitrary dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimension along which to reduce
 *
 * \return A scalar expression (with a potential batch dimension)
 */
Expression std_dim(const Expression& x, unsigned d);

/**
 * \ingroup flowoperations
 * \brief Compute moment along a specific dimension
 * \details Compute the moment of order \f$r\f$, \f$\frac 1 n\sum_{i=1}^nx_i^r\f$ along a specific dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimension along which to reduce
 * \param r Order of the moment
 *
 * \return An expression with one less dimension
 */
Expression moment_dim(const Expression& x, unsigned d, unsigned r);
/**
 * \ingroup flowoperations
 * \brief Compute mean along  a specific dimension
 * \details Computes \f$\frac 1 n\sum_{i=1}^nx_i\f$ along a specific dimension
 *
 * \param x The input mini-batched expression
 * \param d Dimension along which to reduce
 *
 * \return An expression with one less dimension
 */
Expression mean_dim(const Expression& x, unsigned d);


/**
 * \ingroup flowoperations
 * \brief Pick element
 * \details Pick a single element/row/column/sub-tensor from an expression.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param v The index of the element to select
 * \param d The dimension along which to choose the element
 *
 * \return The value of x[v] along dimension d
 */
Expression pick(const Expression& x, unsigned v, unsigned d = 0);

/**
 * \ingroup flowoperations
 * \brief Batched pick
 * \details Pick elements from multiple batches.
 *
 * \param x The input expression
 * \param v A vector of indicies to choose, one for each batch in the
 *          input expression.
 * \param d The dimension along which to choose the elements
 *
 * \return A mini-batched expression containing the picked elements
 */
Expression pick(const Expression& x, const std::vector<unsigned> & v, unsigned d = 0);

/**
 * \ingroup flowoperations
 * \brief Modifiable pick element
 * \details Pick a single element from an expression, where the index is
 *          passed by pointer so we do not need to re-create the computation
 *          graph every time.
 *
 * \param x The input expression
 * \param pv Pointer to the index of the element to select
 * \param d The dimension along which to choose the elements
 *
 * \return The value of x[*pv]
 */
Expression pick(const Expression& x, const unsigned * pv, unsigned d = 0);

/**
 * \ingroup flowoperations
 * \brief Modifiable batched pick element
 * \details Pick multiple elements from an input expression, where the indices
 *          are passed by pointer so we do not need to re-create the computation
 *          graph every time.
 *
 * \param x The input expression
 * \param pv A pointer to vector of indicies to choose
 * \param d The dimension along which to choose the elements
 *
 * \return A mini-batched expression containing the picked elements
 */
Expression pick(const Expression& x, const std::vector<unsigned> * pv, unsigned d = 0);

/**
 * \ingroup flowoperations
 * \brief Pick range of elements
 * \details Pick a range of elements from an expression.
 *
 * \param x The input expression
 * \param s The start index
 * \param e The end index
 * \param d The dimension along which to pick
 *
 * \return The value of {x[v],...,x[u]}
 */
Expression pick_range(const Expression& x, unsigned s, unsigned e, unsigned d = 0);
// DEPRECATED
Expression pickrange(const Expression& x, unsigned s, unsigned e);

/**
 * \ingroup flowoperations
 * \brief (Modifiable) Pick batch element.
 * \details Pick batch element from a batched expression. For a Tensor with 3 batch elements:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1,1} & x_{1,1,2} \\
 *        x_{1,2,1} & x_{1,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 * 
 * pick_batch_elem(t, 1) will return a Tensor of
 * 
 *    \f$
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\ 
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * \param x The input expression
 * \param v The index of the batch element to be picked.
 *
 * \return The expression of picked batch element. The picked element is a tensor
 *         whose `bd` equals to one.
 */
Expression pick_batch_elem(const Expression& x, unsigned v);

/**
 * \ingroup flowoperations
 * \brief (Modifiable) Pick batch elements.
 * \details Pick several batch elements from a batched expression. For a Tensor with 3 batch elements:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1,1} & x_{1,1,2} \\
 *        x_{1,2,1} & x_{1,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 * 
 * pick_batch_elems(t, {2, 3}) will return a Tensor of with 2 batch elements:
 * 
 *    \f$
 *      \begin{pmatrix}
 *        x_{2,1,1} & x_{2,1,2} \\ 
 *        x_{2,2,1} & x_{2,2,2} \\
 *      \end{pmatrix}
 *      \begin{pmatrix}
 *        x_{3,1,1} & x_{3,1,2} \\
 *        x_{3,2,1} & x_{3,2,2} \\
 *      \end{pmatrix}
 *    \f$
 *
 * \param x The input expression
 * \param v A vector of indicies of the batch elements to be picked.
 *
 * \return The expression of picked batch elements. The batch elements is a tensor
 *         whose `bd` equals to the size of vector `v`.
 */
Expression pick_batch_elems(const Expression& x, const std::vector<unsigned> & v);

/**
 * \ingroup flowoperations
 * \brief Pick batch element.
 * \details Pick batch element from a batched expression. 
 * \param x The input expression
 * \param v A pointer to the index of the correct element to be picked.
 *
 * \return The expression of picked batch element. The picked element is a tensor
 *         whose `bd` equals to one.
 */
Expression pick_batch_elem(const Expression& x, const unsigned* v);

/**
 * \ingroup flowoperations
 * \brief Pick batch elements.
 * \details Pick several batch elements from a batched expression. 
 * \param x The input expression
 * \param v A pointer to the indexes
 *
 * \return The expression of picked batch elements. The batch elements is a tensor
 *         whose `bd` equals to the size of vector `v`.
 */
Expression pick_batch_elems(const Expression& x, const std::vector<unsigned> * pv);

/**
 * \ingroup flowoperations
 * \brief Concatenate list of expressions to a single batched expression
 * \details Perform a concatenation of several expressions along the batch dimension.
 *          All expressions must have the same shape except for the batch dimension.
 *
 * \param xs The input expressions
 *
 * \return The expression with the batch dimensions concatenated
 */
inline Expression concatenate_to_batch(const std::initializer_list<Expression>& xs) { return detail::f<ConcatenateToBatch>(xs); }
template <typename T>
inline Expression concatenate_to_batch(const T& xs) { return detail::f<ConcatenateToBatch>(xs); }

/**
 * \ingroup flowoperations
 * \brief Concatenate columns
 * \details Perform a concatenation of the columns in multiple expressions.
 *          All expressions must have the same number of rows.
 *
 * \param xs The input expressions
 *
 * \return The expression with the columns concatenated
 */
inline Expression concatenate_cols(const std::initializer_list<Expression>& xs) { return detail::f<Concatenate>(xs, 1); }
template <typename T>
inline Expression concatenate_cols(const T& xs) { return detail::f<Concatenate>(xs, 1); }

/**
 * \ingroup flowoperations
 * \brief Concatenate
 * \details Perform a concatenation of multiple expressions along
 *          a particular dimension.
 *          All expressions must have the same dimensions except for
 *          the dimension to be concatenated (rows by default).
 *
 * \param xs The input expressions
 * \param xs The dimension along which to perform concatenation
 *
 * \return The expression with the specified dimension concatenated
 */
inline Expression concatenate(const std::initializer_list<Expression>& xs, unsigned d = 0) { return detail::f<Concatenate>(xs, d); }
template <typename T>
inline Expression concatenate(const T& xs, unsigned d = 0) { return detail::f<Concatenate>(xs, d); }

/**
 * \ingroup flowoperations
 * \brief Max out through a dimension
 * \details Select out a element/row/column/sub-tensor from an expression, 
 *          with maximum value along a given dimension.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param d The dimension along which to choose the element
 *
 * \return An expression of sub-tensor with max value along dimension d
 */
Expression max_dim(const Expression& x, unsigned d = 0);

/**
 * \ingroup flowoperations
 * \brief Min out through a dimension
 * \details Select out a element/row/column/sub-tensor from an expression, 
 *          with minimum value along a given dimension.
 *          This will result in the dimension of the tensor being reduced
 *          by 1.
 *
 * \param x The input expression
 * \param d The dimension along which to choose the element
 *
 * \return An expression of sub-tensor with min value along dimension d
 */
Expression min_dim(const Expression& x, unsigned d = 0);


////////////////////////////////////////////////
// Noise operations                           //
////////////////////////////////////////////////

/**
 * \ingroup noiseoperations
 * \brief Gaussian noise
 * \details Add gaussian noise to an expression.
 *
 * \param x The input expression
 * \param stddev The standard deviation of the gaussian
 *
 * \return The noised expression
 */
Expression noise(const Expression& x, real stddev);

/**
 * \ingroup noiseoperations
 * \brief Dropout
 * \details
 *   With a fixed probability, drop out (set to zero) nodes in the input
 *   expression, and **scale** the remaining nodes by 1/p. Note that there are
 *   [two kinds of dropout](http://cs231n.github.io/neural-networks-2/#reg):
 *   - *Regular dropout:* where we perform dropout at training time and then\n
 *     scale outputs by p at test time.
 *   - *Inverted dropout:* where we perform dropout and scaling at training\n
 *     time, and do not need to do anything at test time.
 *   DyNet implements the latter, so you only need to apply dropout at training
 *   time, and do not need to perform scaling and test time.
 *
 * \param x The input expression
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
Expression dropout(const Expression& x, real p);

/**
 * \ingroup noiseoperations
 * \brief Dropout along a specific dimension
 * \details Identical to the dropout operation except the dropout mask is the same across one dimension. Use this if you want to drop columns or lines in a matrix for example 
 * 
 * For now this only supports tensors of order <= 3 (with or without batch dimension)
 *
 * \param x The input expression
 * \param d The dimension along which to drop
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
Expression dropout_dim(const Expression& x, unsigned d, real p);

/**
 * \ingroup noiseoperations
 * \brief Dropout entire elements of a minibatch
 * \details Identical to the dropout operation except entire batch elements are dropped
 * 
 * \param x The input expression
 * \param p The dropout probability
 *
 * \return The dropped out expression
 */
Expression dropout_batch(const Expression& x, real p);

/**
 * \ingroup noiseoperations
 * \brief Block dropout
 * \details Identical to the dropout operation, but either drops out *all*
 *          or *no* values in the expression, as opposed to making a decision
 *          about each value individually.
 *
 * \param x The input expression
 * \param p The block dropout probability
 *
 * \return The block dropout expression
 */
Expression block_dropout(const Expression& x, real p);

////////////////////////////////////////////////
// Convolution operations                     //
////////////////////////////////////////////////

//Expression conv1d_narrow(const Expression& x, const Expression& f);
//Expression conv1d_wide(const Expression& x, const Expression& f);
Expression filter1d_narrow(const Expression& x, const Expression& f);
Expression kmax_pooling(const Expression& x, unsigned k, unsigned d = 1);
Expression fold_rows(const Expression& x, unsigned nrows = 2);
Expression sum_dim(const Expression& x, unsigned d);
Expression sum_cols(const Expression& x);
Expression sum_rows(const Expression& x);
Expression average_cols(const Expression& x);
Expression kmh_ngram(const Expression& x, unsigned n);


/**
 * \ingroup convolutionoperations
 * \brief conv2d without bias
 * \details
 *   2D convolution operator without bias parameters.
 *   'VALID' and 'SAME' convolutions are supported.
 *   Think about when stride is 1, the distinction:
 *   - *SAME*: output size is the same with input size. To do so, one needs to pad the input so the filter can sweep outside of the input maps.
 *   - *VALID*: output size shrinks by filter_size - 1, and the filters always sweep at valid positions inside the input maps. No padding needed.
 *
 *   In detail, assume:
 *   - Input feature maps: (XH x XW x XC) x N
 *   - Filters: FH x FW x XC x FC, 4D tensor
 *   - Strides: strides[0] and strides[1] are row (h) and col (w) stride, respectively.
 *
 *   For the *SAME* convolution: the output height (YH) and width (YW) are computed as:
 *   - YH = ceil(float(XH) / float(strides[0]))
 *   - YW = ceil(float(XW) / float(strides[1]))
 *   and the paddings are computed as:
 *   - pad_along_height = max((YH - 1) * strides[0] + FH - XH, 0)
 *   - pad_along_width = max((YW - 1) * strides[1] + FW - XW, 0)
 *   - pad_top = pad_along_height / 2
 *   - pad_bottom = pad_along_height - pad_top
 *   - pad_left = pad_along_width / 2
 *   - pad_right = pad_along_width - pad_left
 *
 *   For the *VALID* convolution: the output height (YH) and width (YW) are computed as:
 *   - YH = ceil(float(XH - FH + 1) / float(strides[0]))
 *   - YW = ceil(float(XW - FW + 1) / float(strides[1]))
 *   and the paddings are always zeros.
 *
 * \param x The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with an optional batch dimension
 * \param f 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor
 * \param stride the row and column strides
 * \param is_valid 'VALID' convolution or 'SAME' convolution, default is True ('VALID')
 *
 * \return The output feature maps (H x W x Co) x N, 3D tensor with an optional batch dimension
 */
Expression conv2d(const Expression& x, const Expression& f, const std::vector<unsigned>& stride, bool is_valid = true);

/**
 * \ingroup convolutionoperations
 * \brief conv2d with bias
 * \details
 *   2D convolution operator with bias parameters.
 *   'VALID' and 'SAME' convolutions are supported.
 *   Think about when stride is 1, the distinction:
 *   - *SAME*: output size is the same with input size. To do so, one needs to pad the input so the filter can sweep outside of the input maps.
 *   - *VALID*: output size shrinks by filter_size - 1, and the filters always sweep at valid positions inside the input maps. No padding needed.
 *
 *   In detail, assume:
 *   - Input feature maps: XH x XW x XC x N
 *   - Filters: FH x FW x XC x FC 
 *   - Strides: strides[0] and strides[1] are row (h) and col (w) stride, respectively.
 *
 *   For the *SAME* convolution: the output height (YH) and width (YW) are computed as:
 *   - YH = ceil(float(XH) / float(strides[0]))
 *   - YW = ceil(float(XW) / float(strides[1]))
 *   and the paddings are computed as:
 *   - pad_along_height = max((YH - 1) * strides[0] + FH - XH, 0)
 *   - pad_along_width = max((YW - 1) * strides[1] + FW - XW, 0)
 *   - pad_top = pad_along_height / 2
 *   - pad_bottom = pad_along_height - pad_top
 *   - pad_left = pad_along_width / 2
 *   - pad_right = pad_along_width - pad_left
 *
 *   For the *VALID* convolution: the output height (YH) and width (YW) are computed as:
 *   - YH = ceil(float(XH - FH + 1) / float(strides[0]))
 *   - YW = ceil(float(XW - FW + 1) / float(strides[1]))
 *   and the paddings are always zeros.
 *
 * \param x The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with an optional batch dimension
 * \param f 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor
 * \param b The bias (1D: Ci)
 * \param stride the row and column strides
 * \param is_valid 'VALID' convolution or 'SAME' convolution, default is True ('VALID')
 *
 * \return The output feature maps (H x W x Co) x N, 3D tensor with an optional batch dimension
 */
Expression conv2d(const Expression& x, const Expression& f, const Expression& b, const std::vector<unsigned>& stride, bool is_valid = true);

////////////////////////////////////////////////
// Tensor operations                          //
////////////////////////////////////////////////

/**
 * \ingroup tensoroperations
 * \brief Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor
 * \details The resulting tensor \f$z\f$ has coordinates \f$z_ij = \sum_k x_{ijk} y_k\f$
 * 
 * \param x Rank 3 tensor
 * \param y Vector
 * 
 * \return Matrix
 */
Expression contract3d_1d(const Expression& x, const Expression& y);
// z_i = x_ijk * y_k * z_j (+ b_i)
/**
 * \ingroup tensoroperations
 * \brief Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor
 * \details This is the equivalent of calling `contract3d_1d` and then performing a matrix vector multiplication.
 * 
 * The resulting tensor \f$t\f$ has coordinates \f$t_i = \sum_{j,k} x_{ijk} y_k z_j\f$
 * 
 * \param x Rank 3 tensor
 * \param y Vector
 * \param z Vector
 * \return Vector
 */
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z);
/**
 * \ingroup tensoroperations
 * \brief Same as `contract3d_1d_1d` with an additional bias parameter
 * \details This is the equivalent of calling `contract3d_1d` and then performing an affine transform.
 * 
 * The resulting tensor \f$t\f$ has coordinates \f$t_i = b_i + \sum_{j,k} x_{ijk} y_k z_j\f$
 * 
 * \param x Rank 3 tensor
 * \param y Vector
 * \param z Vector
 * \param b Bias vector
 * \return Vector
 */
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b);
// z_ij = x_ijk * y_k + b_ij
/**
 * \ingroup tensoroperations
 * \brief Same as `contract3d_1d` with an additional bias parameter
 * \details The resulting tensor \f$z\f$ has coordinates \f$z_{ij} = b_{ij}+\sum_k x_{ijk} y_k\f$
 * 
 * \param x Rank 3 tensor
 * \param y Vector
 * \param b Bias matrix
 * \return Matrix
 */
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);


////////////////////////////////////////////////
// Linear algebra operations                  //
////////////////////////////////////////////////

/**
 * \ingroup linalgoperations
 * \brief Matrix Inverse
 * \details Takes the inverse of a matrix (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *          Note that back-propagating through an inverted matrix can also be the
 *          source of stability problems sometimes.
 *
 * \param x A square matrix
 *
 * \return The inverse of the matrix
 */
Expression inverse(const Expression& x);

/**
 * \ingroup linalgoperations
 * \brief Log determinant
 * \details Takes the log of the determinant of a matrix.
 *          (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *
 * \param x A square matrix
 *
 * \return The log of its determinant
 */
Expression logdet(const Expression& x);

/**
 * \ingroup linalgoperations
 * \brief Trace of Matrix Product
 * \details Takes the trace of the product of matrices.
 *          (not implemented on GPU yet, although
 *          contributions are welcome: https://github.com/clab/dynet/issues/158).
 *
 * \param x1 A matrix
 * \param x2 Another matrix
 *
 * \return trace(x1 * x2)
 */
Expression trace_of_product(const Expression& x, const Expression& y);

////////////////////////////////////////////////
// Normalization operations                   //
////////////////////////////////////////////////

/**
 * \ingroup normoperations
 * \brief Layer normalization
 * \details Performs layer normalization : 
 * 
 * \f$
 * \begin{split}
 *    \mu &= \frac 1 n \sum_{i=1}^n x_i\\
 *    \sigma &= \sqrt{\frac 1 n \sum_{i=1}^n (x_i-\mu)^2}\\
 *    y&=\frac {\boldsymbol{g}} \sigma \circ (\boldsymbol{x}-\mu) + \boldsymbol{b}\\
 * \end{split}
 * \f$
 * 
 * Reference : [Ba et al., 2016](http://arxiv.org/abs/1607.06450)
 * 
 * \param x Input expression (possibly batched)
 * \param g Gain (same dimension as x, no batch dimension)
 * \param b Bias (same dimension as x, no batch dimension)
 * \return An expression of the same dimension as `x`
 */
Expression layer_norm(const Expression& x, const Expression& g, const Expression& b);
}
// Because expressions are now such a fundamental part of DyNet it doesn't
// make much sense to keep them in separate namespaces, so we import expr
// to the dynet namespace.
using namespace expr;
}

#endif
