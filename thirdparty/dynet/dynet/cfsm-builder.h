#ifndef DYNET_CFSMBUILDER_H
#define DYNET_CFSMBUILDER_H

#include <vector>
#include <string>

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/io-macros.h"

namespace dynet {

class SoftmaxBuilder {
public:
  virtual ~SoftmaxBuilder();

  // call this once per ComputationGraph
  virtual void new_graph(ComputationGraph& cg) = 0;

  // -log(p(w | rep))
  virtual expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx) = 0;

  // samples a word from p(w | rep)
  virtual unsigned sample(const expr::Expression& rep) = 0;

  // returns an Expression representing a vector the size of the vocabulary.
  // The ith dimension gives log p(w_i | rep). This function may be SLOW. Avoid if possible.
  virtual expr::Expression full_log_distribution(const expr::Expression& rep) = 0;

  DYNET_SERIALIZE_COMMIT_EMPTY()
};

class StandardSoftmaxBuilder : public SoftmaxBuilder {
public:
  StandardSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, Model& model);
  void new_graph(ComputationGraph& cg);
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);
  unsigned sample(const expr::Expression& rep);
  expr::Expression full_log_distribution(const expr::Expression& rep);

private:
  StandardSoftmaxBuilder();
  Parameter p_w;
  Parameter p_b;
  expr::Expression w;
  expr::Expression b;
  ComputationGraph* pcg;

  DYNET_SERIALIZE_DECLARE()
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class ClassFactoredSoftmaxBuilder : public SoftmaxBuilder {
 public:
  ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict& word_dict,
                              Model& model);

  void new_graph(ComputationGraph& cg);
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);
  unsigned sample(const expr::Expression& rep);
  expr::Expression full_log_distribution(const expr::Expression& rep);
  void initialize_expressions();

 private:
  ClassFactoredSoftmaxBuilder();
  void read_cluster_file(const std::string& cluster_file, Dict& word_dict);

  Dict cdict;
  std::vector<int> widx2cidx; // will be -1 if not present
  std::vector<unsigned> widx2cwidx; // word index to word index inside of cluster
  std::vector<std::vector<unsigned>> cidx2words;
  std::vector<bool> singleton_cluster; // does cluster contain a single word type?

  // parameters
  Parameter p_r2c;
  Parameter p_cbias;
  std::vector<Parameter> p_rc2ws;     // len = number of classes
  std::vector<Parameter> p_rcwbiases; // len = number of classes

  // Expressions for current graph
  inline expr::Expression& get_rc2w(unsigned cluster_idx) {
    expr::Expression& e = rc2ws[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rc2ws[cluster_idx]);
    return e;
  }
  inline expr::Expression& get_rc2wbias(unsigned cluster_idx) {
    expr::Expression& e = rc2biases[cluster_idx];
    if (!e.pg)
      e = expr::parameter(*pcg, p_rcwbiases[cluster_idx]);
    return e;
  }
  ComputationGraph* pcg;
  expr::Expression r2c;
  expr::Expression cbias;
  std::vector<expr::Expression> rc2ws;
  std::vector<expr::Expression> rc2biases;
  DYNET_SERIALIZE_DECLARE()
};
}  // namespace dynet

BOOST_CLASS_EXPORT_KEY(dynet::StandardSoftmaxBuilder)
BOOST_CLASS_EXPORT_KEY(dynet::ClassFactoredSoftmaxBuilder)

#endif
