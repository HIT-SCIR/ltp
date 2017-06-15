#ifndef DYNET_HSMBUILDER_H
#define DYNET_HSMBUILDER_H

#include <vector>
#include <string>
#include <unordered_map>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dynet/io-macros.h"

namespace dynet {

class Cluster {
private:
  std::vector<Cluster*> children;
  std::vector<unsigned> path;
  std::vector<unsigned> terminals;
  std::unordered_map<unsigned, unsigned> word2ind;
  Parameter p_weights;
  Parameter p_bias;
  mutable expr::Expression weights;
  mutable expr::Expression bias;
  bool initialized;
  unsigned rep_dim;
  unsigned output_size;

  expr::Expression predict(expr::Expression h, ComputationGraph& cg) const;
  DYNET_SERIALIZE_DECLARE()

public:
  Cluster();
  Cluster* add_child(unsigned sym);
  void add_word(unsigned word);
  void initialize(Model& model);
  void initialize(unsigned rep_dim, Model& model);

  void new_graph(ComputationGraph& cg);
  unsigned sample(expr::Expression h, ComputationGraph& cg) const;
  expr::Expression neg_log_softmax(expr::Expression h, unsigned r, ComputationGraph& cg) const;

  unsigned get_index(unsigned word) const;
  unsigned get_word(unsigned index) const;
  unsigned num_children() const;
  const Cluster* get_child(unsigned i) const;
  const std::vector<unsigned>& get_path() const;
  expr::Expression get_weights(ComputationGraph& cg) const;
  expr::Expression get_bias(ComputationGraph& cg) const;

  std::string toString() const;
};

// helps with implementation of hierarchical softmax
// read a file with lines of the following format
// CLASSID   word    [freq]
class HierarchicalSoftmaxBuilder : public SoftmaxBuilder {
 public:
  HierarchicalSoftmaxBuilder(unsigned rep_dim,
                              const std::string& cluster_file,
                              Dict& word_dict,
                              Model& model);
  ~HierarchicalSoftmaxBuilder();

  void initialize(Model& model);

  // call this once per ComputationGraph
  void new_graph(ComputationGraph& cg);

  // -log(p(c | rep) * p(w | c, rep))
  expr::Expression neg_log_softmax(const expr::Expression& rep, unsigned wordidx);

  // samples a word from p(w,c | rep)
  unsigned sample(const expr::Expression& rep);

  expr::Expression full_log_distribution(const expr::Expression& rep);

 private:
  Cluster* read_cluster_file(const std::string& cluster_file, Dict& word_dict);
  std::vector<Cluster*> widx2path; // will be NULL if not found
  Dict path_symbols;

  ComputationGraph* pcg;
  Cluster* root;
};
}  // namespace dynet

#endif
