#include "dynet/cfsm-builder.h"
#include "dynet/except.h"

#include <fstream>
#include <iostream>

#include <boost/serialization/vector.hpp>

using namespace std;

namespace dynet {

using namespace expr;

inline bool is_ws(char x) { return (x == ' ' || x == '\t'); }
inline bool not_ws(char x) { return (x != ' ' && x != '\t'); }

SoftmaxBuilder::~SoftmaxBuilder() {}

StandardSoftmaxBuilder::StandardSoftmaxBuilder() {}

StandardSoftmaxBuilder::StandardSoftmaxBuilder(unsigned rep_dim, unsigned vocab_size, Model& model) {
  p_w = model.add_parameters({vocab_size, rep_dim});
  p_b = model.add_parameters({vocab_size}, ParameterInitConst(0.f));
}

void StandardSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  w = parameter(cg, p_w);
  b = parameter(cg, p_b);
}

Expression StandardSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  return pickneglogsoftmax(affine_transform({b, w, rep}), wordidx);
}

unsigned StandardSoftmaxBuilder::sample(const Expression& rep) {
  Expression dist_expr = softmax(affine_transform({b, w, rep}));
  vector<float> dist = as_vector(pcg->incremental_forward(dist_expr));
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

Expression StandardSoftmaxBuilder::full_log_distribution(const Expression& rep) {
  return log(softmax(affine_transform({b, w, rep})));
}

DYNET_SERIALIZE_COMMIT(StandardSoftmaxBuilder, DYNET_SERIALIZE_DERIVED_DEFINE(SoftmaxBuilder, p_w, p_b))
DYNET_SERIALIZE_IMPL(StandardSoftmaxBuilder)

ClassFactoredSoftmaxBuilder::ClassFactoredSoftmaxBuilder() {}

ClassFactoredSoftmaxBuilder::ClassFactoredSoftmaxBuilder(unsigned rep_dim,
                             const std::string& cluster_file,
                             Dict& word_dict,
                             Model& model) {
  read_cluster_file(cluster_file, word_dict);
  const unsigned num_clusters = cdict.size();
  p_r2c = model.add_parameters({num_clusters, rep_dim});
  p_cbias = model.add_parameters({num_clusters}, ParameterInitConst(0.f));
  p_rc2ws.resize(num_clusters);
  p_rcwbiases.resize(num_clusters);
  for (unsigned i = 0; i < num_clusters; ++i) {
    auto& words = cidx2words[i];  // vector of word ids
    const unsigned num_words_in_cluster = words.size();
    if (num_words_in_cluster > 1) {
      // for singleton clusters, we don't need these parameters, so
      // we don't create them
      p_rc2ws[i] = model.add_parameters({num_words_in_cluster, rep_dim});
      p_rcwbiases[i] = model.add_parameters({num_words_in_cluster}, ParameterInitConst(0.f));
    }
  }
}

void ClassFactoredSoftmaxBuilder::new_graph(ComputationGraph& cg) {
  pcg = &cg;
  const unsigned num_clusters = cdict.size();
  r2c = parameter(cg, p_r2c);
  cbias = parameter(cg, p_cbias);
  rc2ws.clear();
  rc2biases.clear();
  rc2ws.resize(num_clusters);
  rc2biases.resize(num_clusters);
}

Expression ClassFactoredSoftmaxBuilder::neg_log_softmax(const Expression& rep, unsigned wordidx) {
  // TODO check that new_graph has been called
  int clusteridx = widx2cidx[wordidx];
  DYNET_ARG_CHECK(clusteridx >= 0,
                          "Word ID " << wordidx << " missing from clusters in ClassFactoredSoftmaxBuilder::neg_log_softmax");
  Expression cscores = affine_transform({cbias, r2c, rep});
  Expression cnlp = pickneglogsoftmax(cscores, clusteridx);
  if (singleton_cluster[clusteridx]) return cnlp;
  // if there is only one word in the cluster, just return -log p(class | rep)
  // otherwise predict word too
  unsigned wordrow = widx2cwidx[wordidx];
  Expression& cwbias = get_rc2wbias(clusteridx);
  Expression& r2cw = get_rc2w(clusteridx);
  Expression wscores = affine_transform({cwbias, r2cw, rep});
  Expression wnlp = pickneglogsoftmax(wscores, wordrow);
  return cnlp + wnlp;
}

unsigned ClassFactoredSoftmaxBuilder::sample(const Expression& rep) {
  // TODO check that new_graph has been called
  Expression cscores = affine_transform({cbias, r2c, rep});
  Expression cdist_expr = softmax(cscores);
  auto cdist = as_vector(pcg->incremental_forward(cdist_expr));
  unsigned c = 0;
  double p = rand01();
  for (; c < cdist.size(); ++c) {
    p -= cdist[c];
    if (p < 0.0) { break; }
  }
  if (c == cdist.size()) --c;
  unsigned w = 0;
  if (!singleton_cluster[c]) {
    Expression& cwbias = get_rc2wbias(c);
    Expression& r2cw = get_rc2w(c);
    Expression wscores = affine_transform({cwbias, r2cw, rep});
    Expression wdist_expr = softmax(wscores);
    auto wdist = as_vector(pcg->incremental_forward(wdist_expr));
    p = rand01();
    for (; w < wdist.size(); ++w) {
      p -= wdist[w];
      if (p < 0.0) { break; }
    }
    if (w == wdist.size()) --w;
  }
  return cidx2words[c][w];
}

Expression ClassFactoredSoftmaxBuilder::full_log_distribution(const Expression& rep) {
  vector<Expression> full_dist(widx2cidx.size());
  Expression cscores = log(softmax(affine_transform({cbias, r2c, rep})));

  for (unsigned i = 0; i < widx2cidx.size(); ++i) {
    if (widx2cidx[i] == -1) {
      // XXX: Should be -inf
      full_dist[i] = input(*pcg, -10000);
    }
  }

  for (unsigned c = 0; c < p_rc2ws.size(); ++c) {
    Expression cscore = pick(cscores, c);
    if (singleton_cluster[c]) {
      for (unsigned i = 0; i < cidx2words[c].size(); ++i) {
        unsigned w = cidx2words[c][i];
        full_dist[w] = cscore;
      }
    }
    else {
      Expression& cwbias = get_rc2wbias(c);
      Expression& r2cw = get_rc2w(c);
      Expression wscores = affine_transform({cwbias, r2cw, rep});
      Expression wdist = softmax(wscores);

      for (unsigned i = 0; i < cidx2words[c].size(); ++i) {
        unsigned w = cidx2words[c][i];
        full_dist[w] = pick(wdist, i) + cscore;
      }
    }
  }

  return log(softmax(concatenate(full_dist)));
}

void ClassFactoredSoftmaxBuilder::read_cluster_file(const std::string& cluster_file, Dict& word_dict) {
  cerr << "Reading clusters from " << cluster_file << " ...\n";
  ifstream in(cluster_file);
  if(!in)
    DYNET_INVALID_ARG("Could not find cluster file " << cluster_file << " in ClassFactoredSoftmax");
  int wc = 0;
  string line;
  while(getline(in, line)) {
    ++wc;
    const unsigned len = line.size();
    unsigned startc = 0;
    while (is_ws(line[startc]) && startc < len) { ++startc; }
    unsigned endc = startc;
    while (not_ws(line[endc]) && endc < len) { ++endc; }
    unsigned startw = endc;
    while (is_ws(line[startw]) && startw < len) { ++startw; }
    unsigned endw = startw;
    while (not_ws(line[endw]) && endw < len) { ++endw; }
    if(endc <= startc || startw <= endc || endw <= startw)
      DYNET_INVALID_ARG("Invalid format in cluster file " << cluster_file << " in ClassFactoredSoftmax");
    unsigned c = cdict.convert(line.substr(startc, endc - startc));
    unsigned word = word_dict.convert(line.substr(startw, endw - startw));
    if (word >= widx2cidx.size()) {
      widx2cidx.resize(word + 1, -1);
      widx2cwidx.resize(word + 1);
    }
    widx2cidx[word] = c;
    if (c >= cidx2words.size()) cidx2words.resize(c + 1);
    auto& clusterwords = cidx2words[c];
    widx2cwidx[word] = clusterwords.size();
    clusterwords.push_back(word);
  }
  singleton_cluster.resize(cidx2words.size());
  int scs = 0;
  for (unsigned i = 0; i < cidx2words.size(); ++i) {
    bool sc = cidx2words[i].size() <= 1;
    if (sc) scs++;
    singleton_cluster[i] = sc;
  }
  cerr << "Read " << wc << " words in " << cdict.size() << " clusters (" << scs << " singleton clusters)\n";
}

DYNET_SERIALIZE_COMMIT(ClassFactoredSoftmaxBuilder,
		       DYNET_SERIALIZE_DERIVED_DEFINE(SoftmaxBuilder, cdict, widx2cidx, widx2cwidx, cidx2words, singleton_cluster, p_r2c, p_cbias, p_rc2ws, p_rcwbiases))

void ClassFactoredSoftmaxBuilder::initialize_expressions() {
  for (unsigned c = 0; c < p_rc2ws.size(); ++c) {
    //get_rc2w(_bias) creates the expression at c if the expression does not already exist.
    get_rc2w(c);
    get_rc2wbias(c);
  }
}

DYNET_SERIALIZE_IMPL(ClassFactoredSoftmaxBuilder)

} // namespace dynet

BOOST_CLASS_EXPORT_IMPLEMENT(dynet::StandardSoftmaxBuilder)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::ClassFactoredSoftmaxBuilder)
