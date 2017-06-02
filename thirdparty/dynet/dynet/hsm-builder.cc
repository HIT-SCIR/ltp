#include "dynet/hsm-builder.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

namespace dynet {

using namespace expr;

Cluster::Cluster() : initialized(false) {}
void Cluster::new_graph(ComputationGraph& cg) {
  for (Cluster* child : children) {
    child->new_graph(cg);
  }
  bias.pg = NULL;
  weights.pg = NULL;
}

Cluster* Cluster::add_child(unsigned sym) {
  auto it = word2ind.find(sym);
  unsigned i;
  if (it == word2ind.end()) {
    Cluster* c = new Cluster();
    c->rep_dim = rep_dim;
    c->path = path;
    c->path.push_back(sym);
    i = children.size();
    word2ind.insert(make_pair(sym, i));
    children.push_back(c);
  }
  else {
    i = it->second;
  }
  return children[i];
}

void Cluster::add_word(unsigned word) {
  word2ind[word] = terminals.size();
  terminals.push_back(word);
}

void Cluster::initialize(unsigned rep_dim, Model& model) {
  this->rep_dim = rep_dim;
  initialize(model);
}

void Cluster::initialize(Model& model) {
  output_size = (children.size() > 0) ? children.size() : terminals.size();

  if (output_size == 1) {
  }
  else if (output_size == 2) {
    p_weights = model.add_parameters({1, rep_dim});
    p_bias = model.add_parameters({1}, ParameterInitConst(0.f));
  }
  else {
    p_weights = model.add_parameters({output_size, rep_dim});
    p_bias = model.add_parameters({output_size}, ParameterInitConst(0.f));
  }

  for (Cluster* child : children) {
    child->initialize(model);
  }
}

unsigned Cluster::num_children() const {
  return children.size();
}

const Cluster* Cluster::get_child(unsigned i) const {
  return children[i];
}

const vector<unsigned>& Cluster::get_path() const { return path; }
unsigned Cluster::get_index(unsigned word) const { return word2ind.find(word)->second; }
unsigned Cluster::get_word(unsigned index) const { return terminals[index]; }

Expression Cluster::predict(Expression h, ComputationGraph& cg) const {
  if (output_size == 1) {
    return input(cg, 1.0f);
  }
  else {
    Expression b = get_bias(cg);
    Expression w = get_weights(cg);
    return affine_transform({b, w, h});
  }
}

Expression Cluster::neg_log_softmax(Expression h, unsigned r, ComputationGraph& cg) const {
  if (output_size == 1) {
    return input(cg, 0.0f);
  }
  else if (output_size == 2) {
    Expression p = logistic(predict(h, cg));
    if (r == 1) {
      p = 1 - p;
    }
    return -log(p);
  }
  else {
    Expression dist = predict(h, cg);
    return pickneglogsoftmax(dist, r);
  }
}

unsigned Cluster::sample(expr::Expression h, ComputationGraph& cg) const {
  if (output_size == 1) {
    return 0;
  }
  else if (output_size == 2) {
    expr::Expression prob0_expr = logistic(predict(h, cg));
    double prob0 = as_scalar(cg.incremental_forward(prob0_expr));
    double p = rand01();
    if (p < prob0) {
      return 0;
    }
    else {
      return 1;
    }
  }
  else {
    expr::Expression dist_expr = softmax(predict(h, cg));
    vector<float> dist = as_vector(cg.incremental_forward(dist_expr));
    unsigned c = 0;
    double p = rand01();
    for (; c < dist.size(); ++c) {
      p -= dist[c];
      if (p < 0.0) { break; }
    }
    if (c == dist.size()) {
      --c;
    }
    return c;
  }
}

Expression Cluster::get_weights(ComputationGraph& cg) const {
  if (weights.pg != &cg) {
    weights = parameter(cg, p_weights);
  }
  return weights;
}

Expression Cluster::get_bias(ComputationGraph& cg) const {
  if (bias.pg != &cg) {
    bias = parameter(cg, p_bias);
  }
  return bias;
}

string Cluster::toString() const {
  stringstream ss;
  for (unsigned i = 0; i < path.size(); ++i) {
    if (i != 0) {
      ss << " ";
    }
    ss << path[i];
  }
  return ss.str();
}

#if BOOST_VERSION >= 105600
  DYNET_SERIALIZE_COMMIT(Cluster, DYNET_SERIALIZE_DEFINE(rep_dim, children, path, terminals, word2ind))
#else
  template<class Archive>
  void Cluster::serialize(Archive& ar, const unsigned int) {
    DYNET_RUNTIME_ERR("Serializing clusters is only supported on versions of boost 1.56 or higher");
  }
#endif
DYNET_SERIALIZE_IMPL(Cluster)

HierarchicalSoftmaxBuilder::HierarchicalSoftmaxBuilder(unsigned rep_dim,
                             const std::string& cluster_file,
                             Dict& word_dict,
                             Model& model) {
  root = read_cluster_file(cluster_file, word_dict);
  root->initialize(rep_dim, model);
}

HierarchicalSoftmaxBuilder::~HierarchicalSoftmaxBuilder() {
}

void HierarchicalSoftmaxBuilder::initialize(Model& model) {
 root->initialize(model);
}

void HierarchicalSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  root->new_graph(cg);
}

Expression HierarchicalSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  if(pcg != NULL)
    DYNET_INVALID_ARG("In HierarchicalSoftmaxBuilder, you must call new_graph before calling neg_log_softmax!");
  Cluster* path = widx2path[wordidx];

  unsigned i = 0;
  const Cluster* node = root;
  DYNET_ASSERT(root != NULL, "Null root in HierarchicalSoftmaxBuilder");
  vector<Expression> log_probs;
  Expression lp;
  unsigned r;
  while (node->num_children() > 0) {
    r = node->get_index(path->get_path()[i]);
    lp = node->neg_log_softmax(rep, r, *pcg);
    log_probs.push_back(lp);
    node = node->get_child(r);
    DYNET_ASSERT(node != NULL, "Null node in HierarchicalSoftmaxBuilder");
    i += 1;
  }

  r = path->get_index(wordidx);
  lp = node->neg_log_softmax(rep, r, *pcg);
  log_probs.push_back(lp);

  return sum(log_probs);
}

unsigned HierarchicalSoftmaxBuilder::sample(const expr::Expression& rep) {
  if(pcg != NULL)
    DYNET_INVALID_ARG("In HierarchicalSoftmaxBuilder, you must call new_graph before calling sample!");

  const Cluster* node = root;
  vector<float> dist;
  unsigned c;
  while (node->num_children() > 0) {
    c = node->sample(rep, *pcg);
    node = node->get_child(c);
  }

  c = node->sample(rep, *pcg);
  return node->get_word(c);
}

Expression HierarchicalSoftmaxBuilder::full_log_distribution(const Expression& rep) {
  DYNET_RUNTIME_ERR("full_distribution not implemented for HierarchicalSoftmaxBuilder");
  return dynet::expr::Expression();
}

inline bool is_ws(char x) { return (x == ' ' || x == '\t'); }
inline bool not_ws(char x) { return (x != ' ' && x != '\t'); }

Cluster* HierarchicalSoftmaxBuilder::read_cluster_file(const std::string& cluster_file, Dict& word_dict) {
  cerr << "Reading clusters from " << cluster_file << " ...\n";
  ifstream in(cluster_file);
  if(!in)
    DYNET_INVALID_ARG("HierarchicalSoftmaxBuilder couldn't read clusters from " << cluster_file);
  int wc = 0;
  string line;
  vector<unsigned> path;
  Cluster* root = new Cluster();
  while(getline(in, line)) {
    path.clear();
    ++wc;
    const unsigned len = line.size();
    unsigned startp = 0;
    unsigned endp = 0;
    while (startp < len) {
      while (is_ws(line[startp]) && startp < len) { ++startp; }
      endp = startp;
      while (not_ws(line[endp]) && endp < len) { ++endp; }
      string symbol = line.substr(startp, endp - startp);
      path.push_back(path_symbols.convert(symbol));
      if (line[endp] == ' ') {
        startp = endp + 1;
        continue;
      }
      else {
        break;
      }
    }
    Cluster* node = root;
    for (unsigned symbol : path) {
      node = node->add_child(symbol);
    }

    unsigned startw = endp;
    while (is_ws(line[startw]) && startw < len) { ++startw; }
    unsigned endw = startw;
    while (not_ws(line[endw]) && endw < len) { ++endw; }
    if(endp <= startp || startw <= endp || endw <= startw)
      DYNET_INVALID_ARG("File formatting error in HierarchicalSoftmaxBuilder");

    string word = line.substr(startw, endw - startw);
    unsigned widx = word_dict.convert(word);
    node->add_word(widx);

    if (widx2path.size() <= widx) {
      widx2path.resize(widx + 1);
    }
    widx2path[widx] = node;
  }
  cerr << "Done reading clusters.\n";
  return root;
}

} // namespace dynet
