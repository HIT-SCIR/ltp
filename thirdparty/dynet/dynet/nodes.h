#ifndef DYNET_NODES_H_
#define DYNET_NODES_H_

#include "dynet/dynet.h"
#include "dynet/devices.h"
#include "dynet/nodes-macros.h"

// See nodes-macros.h for more details about DYNET_NODE_DEFINE_DEV_IMPL().

namespace dynet {

// M = x_0, v = x_1
// y = M + v (broadcasting over columns)
struct AddVectorToAllColumns : public Node {
  explicit AddVectorToAllColumns(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = L_sparsemax(x_0; q)
// where x_0 is a vector of "unnormalized" probabilities
// q are the vector of labels
struct SparsemaxLoss : public Node {
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& target) : Node(a), q(target), pq(&q) {}
  explicit SparsemaxLoss(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* ptarget) : Node(a), q(), pq(ptarget) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  const std::vector<unsigned> q;
  const std::vector<unsigned>* pq;
};

// y = sparsemax(x)
// y = arg min_y ||y - x||^2
struct Sparsemax : public Node {
  explicit Sparsemax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
};

// y = inv(x)
// x = an invertible matrix
struct MatrixInverse : public Node {
  explicit MatrixInverse(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = select_rows(x, rows)
// x = a matrix
struct SelectRows : public Node {
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& r) : Node(a), rows(r), prows(&rows) {}
  explicit SelectRows(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pr) : Node(a), prows(pr) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> rows;
  const std::vector<unsigned>* prows;
};

// y = select_cols(x, cols)
// x = a matrix
struct SelectCols : public Node {
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& c) : Node(a), cols(c), pcols(&cols) {}
  explicit SelectCols(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pc) : Node(a), pcols(pc) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> cols;
  const std::vector<unsigned>* pcols;
};

// y = pow(x_1, x_2)
// x_2 raise every element in x_1 to the power of scalar x_2
struct Pow : public Node {
  explicit Pow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = min{x_1, x_2}
struct Min : public Node {
  explicit Min(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
};

// y = max{x_1, x_2}
struct Max : public Node {
  template <typename T> explicit Max(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
};

// y = Tr(x_1 * x_2^T)
struct TraceOfProduct : public Node {
  explicit TraceOfProduct(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = alpha * x_1
struct ConstScalarMultiply : public Node {
  explicit ConstScalarMultiply(const std::initializer_list<VariableIndex>& a, float alpha) : Node(a), alpha(alpha) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  float alpha;
};

// y = x_1^T . x_2
struct DotProduct : public Node {
  explicit DotProduct(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1^T
// NOTE: if you have a column or row vector as input, runtime is constant
// if you have a matrix as input, the runtime is O(mn) - try to avoid using this
struct Transpose : public Node {
  explicit Transpose(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned> & dims) : Node(a), dims(dims) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  std::vector<unsigned> dims;
};

// y = reshape(x_1, --> to)
struct Reshape : public Node {
  explicit Reshape(const std::initializer_list<VariableIndex>& a, const Dim& to) : Node(a), to(to) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  Dim to;
};

// y_i = \sum_{j=1}^n x_1:{i-1+j}
struct KMHNGram : public Node {
  explicit KMHNGram(const std::initializer_list<VariableIndex>& a, unsigned n) : Node(a), n(n) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned n;  // width, n=2 for Karl's paper
};

// n_{i,j} ~ N(0,stddev)
// y = x + n
struct GaussianNoise : public Node {
  explicit GaussianNoise(const std::initializer_list<VariableIndex>& a, real stddev) : Node(a), stddev(stddev) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  real stddev;
};

// y = dropout(x,p) where p specifies the dropout probability
struct Dropout : public Node {
  explicit Dropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  real p;
};

// y = dropout(x,p) where p specifies the dropout probability
struct DropoutDim : public Node {
  explicit DropoutDim(const std::initializer_list<VariableIndex>& a, unsigned d,real p) : Node(a), dimension(d), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  unsigned dimension;
  real p;
};

// y = dropout(x,p) where p specifies the dropout probability
struct DropoutBatch : public Node {
  explicit DropoutBatch(const std::initializer_list<VariableIndex>& a, real p) : Node(a), p(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
  real p;
};

// y = block_dropout(x,p) where p specifies the probability for dropping-out the entire block
struct BlockDropout : public Node {
  explicit BlockDropout(const std::initializer_list<VariableIndex>& a, real p) : Node(a), dropout_probability(p) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  real dropout_probability;
};

// y = c + x_1
// (c is a vector or matrix of the constant, usually 1, but can be configured)
struct ConstantPlusX : public Node {
  explicit ConstantPlusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  real c;
};

// y = c - x_1
// (c is a vector or matrix of the constant, usually 1, but can be configured)
struct ConstantMinusX : public Node {
  explicit ConstantMinusX(const std::initializer_list<VariableIndex>& a, real o) : Node(a), c(o) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  real c;
};

// y = sqrt x_1
struct Sqrt : public Node {
  explicit Sqrt(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = abs x_1
struct Abs : public Node {
  explicit Abs(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = erf x_1
struct Erf : public Node {
  explicit Erf(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = tanh x_1
struct Tanh : public Node {
  explicit Tanh(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \odot x_1
struct Square : public Node {
  explicit Square(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \odot x_1 \odot x_1
struct Cube : public Node {
  explicit Cube(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = exp x_1
struct Exp : public Node {
  explicit Exp(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = lgamma x_1
struct LogGamma : public Node {
  explicit LogGamma(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = log x_1  (base e, i.e., natural log)
struct Log : public Node {
  explicit Log(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// concatenate rows
struct Concatenate : public Node {
  template <typename T> explicit Concatenate(const T& a, unsigned d) : Node(a), dimension(d) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  // src_row_indices[i] says what row in fx the ith x vector was assigned to
  // used to simplify backprop
  mutable std::vector<unsigned> src_indices;
  unsigned dimension;
};

// concatenate different batched experssions into one single batched tensor
struct ConcatenateToBatch : public Node {
  template <typename T> explicit ConcatenateToBatch(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override {return true;}
  mutable std::vector<unsigned> src_element_indices;
};

// x_1 is a scalar (or row vector)
// x_2 is a scalar (or row vector)
// y = max(0, margin - x_1 + x_2)
struct PairwiseRankLoss : public Node {
  explicit PairwiseRankLoss(const std::initializer_list<VariableIndex>& a, real m = 1.0) : Node(a), margin(m) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  real margin;
};

// Let x be a vector-valued input, x_i represents the score of the ith element, then
// y = \sum{i != element} max{0, margin - x_element + x_i}
struct Hinge : public Node {
  explicit Hinge(const std::initializer_list<VariableIndex>& a, unsigned e, real m = 1.0) : Node(a), element(e), pelement(&element), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const unsigned* pe, real m = 1.0) : Node(a), element(), pelement(pe), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& e, real m = 1.0) : Node(a), element(), pelement(), elements(e), pelements(&elements), margin(m) {}
  explicit Hinge(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pe, real m = 1.0) : Node(a), element(), pelement(), elements(), pelements(pe), margin(m) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  unsigned element;
  const unsigned* pelement;
  std::vector<unsigned> elements;
  const std::vector<unsigned>* pelements;
  real margin;
};

// y = x_1, but dy/dx is set to 0
struct NoBackprop : public Node {
  explicit NoBackprop(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1, dy/dx is set to negative. 
struct FlipGradient : public Node {
  explicit FlipGradient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};  
  
// y = x_1
struct Identity : public Node {
  explicit Identity(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// hyperparameter: width > 1
// x_1 is a vector in R^n, which we write x
// y is a vector in R^{n / width}
// y_i = max_{x_{i * width - width + 1}, ..., x_{i * width}}
struct MaxPooling1D : public Node {
  MaxPooling1D(const std::initializer_list<VariableIndex>& a, unsigned w) : Node(a), width(w) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  unsigned width;
  mutable std::vector<unsigned> ind;
};

// y = x_1 * x_2
struct MatrixMultiply : public Node {
  explicit MatrixMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \cdot x_2  (Hadamard product)
struct CwiseMultiply : public Node {
  explicit CwiseMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 + x_2  (Addition where x_2 is a scalar)
struct ScalarAdd : public Node {
  explicit ScalarAdd(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \cdot x_2  (Hadamard product where x_1 is a scalar)
struct ScalarMultiply : public Node {
  explicit ScalarMultiply(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 / x_2  (Elementwise division where x_2 is a scalar)
struct ScalarQuotient : public Node {
  explicit ScalarQuotient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 / x_2  (cwiseQuotient)
struct CwiseQuotient : public Node {
  explicit CwiseQuotient(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x_1 \sum_{i=2, 4 ...} A_i * x_{i+1}
struct AffineTransform : public Node {
  template <typename T> explicit AffineTransform(const T& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
  mutable float* dEdf_mem;
};

// y = -x_1
struct Negate : public Node {
  explicit Negate(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; } 
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = max(0,x)
struct Rectify : public Node {
  explicit Rectify(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// you could do this with LogisticSigmoid, Softmax or a variety of other
// functions, but this is often useful.
// x_1 must be a vector with values between 0 and 1
// target_y is an equivalently sized vector w values between 0 and 1
// y = ty * log(x_1) + (1 - ty) * log(x_1)
struct BinaryLogLoss : public Node {
  BinaryLogLoss(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = \log \sum_i \exp x_i
// done in log space carefully to avoid over/underflow issues
struct LogSumExp : public Node {
  template <typename T> explicit LogSumExp(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
};

struct LogDet : public Node {
  template <typename T> explicit LogDet(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = \sum_i x_i
struct Sum : public Node {
  template <typename T> explicit Sum(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i,j,... x[i,j,...]
struct SumElements : public Node {
  template <typename T> explicit SumElements(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i x_i
struct SumBatches : public Node {
  template <typename T> explicit SumBatches(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i,j,... x[i,j,...]
struct StdElements : public Node {
  template <typename T> explicit StdElements(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// y = \sum_i x_i
struct StdBatches : public Node {
  template <typename T> explicit StdBatches(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

//y = \sum_i x_i
struct StdDimension : public Node {
  template <typename T> explicit StdDimension(const T& a, unsigned d) : Node(a), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned dimension;
};

// y = \sum_i,j,... x[i,j,...]
struct MomentElements : public Node {
  template <typename T> explicit MomentElements(const T& a, unsigned o) : Node(a), order(o) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned order;
};

// y = \sum_i x_i
struct MomentBatches : public Node {
  template <typename T> explicit MomentBatches(const T& a, unsigned o) : Node(a), order(o) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned order;
};

//y = \sum_i x_i
struct MomentDimension : public Node {
  template <typename T> explicit MomentDimension(const T& a, unsigned d, unsigned o) : Node(a), dimension(d), order(o) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
private:
  unsigned dimension;
  unsigned order;
};

// y = ( \sum_i x_i ) / |x|
struct Average : public Node {
  template <typename T> explicit Average(const T& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
};

// this is used to implement poisson regression
// x_1 = log predicted mean
// ty = true y (this is not a VariableIndex since it has to be a nonnegative integer and
//              is therefore nondifferentiable. There are various continuous extensions
//              using the incomplete gamma function that could be used, but meh)
// y = log Poisson(ty; \lambda = \exp x_1)
//   = ty*x_1 - exp(x_1) - log(ty!)
struct PoissonRegressionLoss : public Node {
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, unsigned true_y) : Node(a), ty(true_y), pty(&ty) {}
  explicit PoissonRegressionLoss(const std::initializer_list<VariableIndex>& a, const unsigned* ptrue_y) : Node(a), ty(), pty(ptrue_y) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
 private:
  unsigned ty;
  const unsigned* pty;
};

// y = || x_1 ||^2
struct SquaredNorm : public Node {
  explicit SquaredNorm(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 ||
struct L2Norm : public Node {
  explicit L2Norm(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 - x_2 ||^2
struct SquaredEuclideanDistance : public Node {
  explicit SquaredEuclideanDistance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = || x_1 - x_2 ||_H(d)
struct HuberDistance : public Node {
  explicit HuberDistance(const std::initializer_list<VariableIndex>& a, float d = 1.345f) : Node(a), d(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  float d;
};

// y = || x_1 - x_2 ||_1
struct L1Distance : public Node {
  explicit L1Distance(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = \sigma(x_1)
struct LogisticSigmoid : public Node {
  explicit LogisticSigmoid(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// y = x / (1 + |x|)
struct SoftSign : public Node {
  explicit SoftSign(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  virtual bool supports_multibatch() const override { return true; }
  DYNET_NODE_DEFINE_DEV_IMPL()
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i / z
struct Softmax : public Node {
  explicit Softmax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
};

// z = \sum_j \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct LogSoftmax : public Node {
  explicit LogSoftmax(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  size_t aux_storage_size() const override;
  virtual bool supports_multibatch() const override { return true; }
};

// z = \sum_j \exp (x_i)_j
// y = (x_1)_element - \log z
struct PickNegLogSoftmax : public Node {
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  // use this constructor if you want to perform mini-batching
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickNegLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};

// z = \sum_{j \in denom} \exp (x_i)_j
// y_i = (x_1)_i - \log z
struct RestrictedLogSoftmax : public Node {
  explicit RestrictedLogSoftmax(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& d) : Node(a), denom(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  std::vector<unsigned> denom;
};

// x_1 is a vector
// y = (x_1)_{*pval}
// this is used to implement cross-entropy training
struct PickElement : public Node {
  explicit PickElement(const std::initializer_list<VariableIndex>& a, unsigned v, unsigned d = 0) : Node(a), val(v), pval(&val), vals(), pvals(), dimension(d) {}
  // use this constructor if you want to perform mini-batching
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v, unsigned d = 0) : Node(a), val(), pval(), vals(v), pvals(&vals), dimension(d) {}
  // use these constructors if you want to change the value after the graph is constructed
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const unsigned* pv, unsigned d = 0) : Node(a), val(), pval(pv), vals(), pvals(), dimension(d) {}
  explicit PickElement(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv, unsigned d = 0) : Node(a), val(), pval(), vals(), pvals(pv), dimension(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
  unsigned dimension;
};

// x_1 is a tensor
// y = x_1[start:end] along dimension d
// (start inclusive, end exclusive)
struct PickRange : public Node {
  explicit PickRange(const std::initializer_list<VariableIndex>& a, unsigned s, unsigned e, unsigned d = 0) : Node(a), start(s), end(e), dim(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned start, end, dim;
};

// x is a batched tensor
// y = (x)_{[*pval]}
struct PickBatchElements : public Node {
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, unsigned v) : Node(a), val(v), pval(&val), vals(), pvals() {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>& v) : Node(a), val(), pval(), vals(v), pvals(&vals) {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const unsigned* pv) : Node(a), val(), pval(pv), vals(), pvals() {}
  explicit PickBatchElements(const std::initializer_list<VariableIndex>& a, const std::vector<unsigned>* pv) : Node(a), val(), pval(), vals(), pvals(pv) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  unsigned val;
  const unsigned* pval;
  std::vector<unsigned> vals;
  const std::vector<unsigned>* pvals;
};

// represents a simple vector of 0s
struct Zeroes : public Node {
  explicit Zeroes(const Dim& d) : dim(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
};

// draw random noise from Normal(0, 1)
struct RandomNormal : public Node {
  explicit RandomNormal(const Dim& d) : dim(d) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
};

// draw from Bernoulli(p)
struct RandomBernoulli : public Node {
  explicit RandomBernoulli(const std::initializer_list<VariableIndex>& a, const Dim& d, real p, real scale = 1.0f) : dim(d), p(p), scale(scale) {
    DYNET_ASSERT(a.size() == 0, "RandomBernoulli doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real p;
  real scale;
};

// draw a random real from Uniform(left, right)
struct RandomUniform : public Node {
  explicit RandomUniform(const std::initializer_list<VariableIndex>& a, const Dim& d, real left, real right) : dim(d), left(left), right(right) {
    DYNET_ASSERT(a.size() == 0, "RandomUniform doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real left, right;
};

// draw a random real from Uniform(left, right)
struct RandomGumbel : public Node {
  explicit RandomGumbel(const std::initializer_list<VariableIndex>& a, const Dim& d, real mu, real beta) : dim(d), mu(mu), beta(beta) {
    DYNET_ASSERT(a.size() == 0, "RandomGumbel doesn't accept nodes as input");
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  Dim dim;
  real mu, beta;
};

struct MaxDimension : public Node {
  explicit MaxDimension(const std::initializer_list<VariableIndex>& a, unsigned dimension = 0) : Node(a), reduced_dim(dimension) {
    first_dim = reduced_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == reduced_dim ? first_dim + 2 : first_dim + 1;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned reduced_dim;
  unsigned first_dim;
  unsigned second_dim;
};

struct MinDimension : public Node {
  explicit MinDimension(const std::initializer_list<VariableIndex>& a, unsigned dimension = 0) : Node(a), reduced_dim(dimension) {
    first_dim = reduced_dim == 0 ? 1 : 0;
    second_dim = first_dim + 1 == reduced_dim ? first_dim + 2 : first_dim + 1;
  }
  DYNET_NODE_DEFINE_DEV_IMPL()
  virtual bool supports_multibatch() const override { return true; }
  size_t aux_storage_size() const override;
  unsigned reduced_dim;
  unsigned first_dim;
  unsigned second_dim;
};

} // namespace dynet

#endif
