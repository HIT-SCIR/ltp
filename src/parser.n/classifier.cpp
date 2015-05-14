#include "parser.n/classifier.h"
#include "utils/logging.hpp"

namespace ltp {
namespace depparser {

Sample::Sample() {}

Sample::Sample(const std::vector<int>& _attributes,
    const std::vector<double>& _classes)
  : attributes(_attributes), classes(_classes) {
}


NeuralNetworkClassifier::NeuralNetworkClassifier()
  : initialized(false), embedding_size(0),
  hidden_layer_size(0), nr_objects(0), nr_feature_types(0), nr_classes(0) {}

void NeuralNetworkClassifier::initialize(
    int _nr_objects,
    int _nr_classes,
    int _nr_feature_types,
    const LearnOption& opt,
    const std::vector< std::vector<double> >& embeddings,
    const std::vector<int>& precomputed_features
    ) {
  if (initialized) {
    ERROR_LOG("classifier: weight should not be initialized twice!");
    return;
  }

  batch_size = opt.batch_size;
  fix_embeddings = opt.fix_embeddings;
  dropout_probability = opt.dropout_probability;
  lambda = opt.lambda;
  ada_eps = opt.ada_eps;
  ada_alpha = opt.ada_alpha;

  // Initialize the parameter.
  nr_feature_types = _nr_feature_types;
  nr_objects = _nr_objects;
  nr_classes = _nr_classes; // nr_deprels*2+1-NIL

  embedding_size = opt.embedding_size;
  hidden_layer_size = opt.hidden_layer_size;

  // Initialize the network
  int nrows = hidden_layer_size;
  int ncols = embedding_size * nr_feature_types;
  W1 = (2.* arma::randu<arma::mat>(nrows, ncols)- 1.) * sqrt(6./ (nrows+ ncols));
  b1 = (2.* arma::randu<arma::vec>(nrows)- 1.) * sqrt(6./ (nrows+ ncols));

  nrows = _nr_classes;  //
  ncols = hidden_layer_size;
  W2 = (2.* arma::randu<arma::mat>(nrows, ncols)- 1.) * sqrt(6./ (nrows+ ncols));

  // Initialized the embedding
  nrows = embedding_size;
  ncols= _nr_objects;

  E = (2.* arma::randu<arma::mat>(nrows, ncols) - 1.) * opt.init_range;
  for (size_t i = 0; i < embeddings.size(); ++ i) {
    const std::vector<double>& embedding = embeddings[i];
    int id = embedding[0];
    for (unsigned j = 1; j < embedding.size(); ++ j) {
      E(j-1, id) = embedding[j];
    }
  }

  grad_W1 = arma::zeros<arma::mat>(W1.n_rows, W1.n_cols);
  grad_b1 = arma::zeros<arma::vec>(b1.n_rows);
  grad_W2 = arma::zeros<arma::mat>(W2.n_rows, W2.n_cols);
  grad_E = arma::zeros<arma::mat>(E.n_rows, E.n_cols);

  // Initialized the precomputed features
  std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  size_t rank = 0;

  for (size_t i = 0; i < precomputed_features.size(); ++ i) {
    int fid = precomputed_features[i];
    encoder[fid] = rank ++;
  }

  saved.zeros(hidden_layer_size, encoder.size());
  grad_saved.zeros(hidden_layer_size, encoder.size());

  //
  initialize_gradient_histories();

  initialized = true;

  info();
  TRACE_LOG("classifier: size of batch = %d", batch_size);
  TRACE_LOG("classifier: alpha = %lf", ada_alpha);
  TRACE_LOG("classifier: eps = %lf", ada_eps);
  TRACE_LOG("classifier: lambda = %lf", lambda);
  TRACE_LOG("classifier: fix embedding = %s", (fix_embeddings? "true": "false"));
}

void NeuralNetworkClassifier::score(const std::vector<int>& attributes,
    std::vector<double>& retval) {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  arma::vec hidden_layer = arma::zeros<arma::vec>(hidden_layer_size);

  for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
    int aid = attributes[i];
    int fid = aid * nr_feature_types + i;
    std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
    if (rep != encoder.end()) {
      hidden_layer += saved.col(rep->second);
    } else {
      // W1[0:hidden_layer, off:off+embedding_size] * E[fid:]'
      hidden_layer += W1.submat(0, off, hidden_layer_size-1, off+embedding_size-1) *
        E.col(aid);
    }
  }

  hidden_layer += b1;
  hidden_layer = hidden_layer % hidden_layer % hidden_layer;

  arma::vec output = W2 * hidden_layer;
  retval.resize(nr_classes, 0.);
  for (int i = 0; i < nr_classes; ++ i) { retval[i] = output(i); }
}

void NeuralNetworkClassifier::compute_ada_gradient_step(
    std::vector<Sample>::const_iterator begin,
    std::vector<Sample>::const_iterator end) {
  if (!initialized) {
    ERROR_LOG("classifier: should not run the learning algorithm"
        " with un-initialized classifier.");
    return;
  }

  // precomputing
  std::unordered_set<int> precomputed_features;
  get_precomputed_features(begin, end, precomputed_features);
  precomputing(precomputed_features);

  // calculate gradient
  grad_saved.zeros();
  compute_gradient(begin, end, end- begin);
  compute_saved_gradient(precomputed_features);

  // add regularizer.
  add_l2_regularization();
}

void NeuralNetworkClassifier::initialize_gradient_histories() {
  eg2E = arma::zeros<arma::mat>(E.n_rows, E.n_cols);
  eg2W1 = arma::zeros<arma::mat>(W1.n_rows, W1.n_cols);
  eg2W2 = arma::zeros<arma::mat>(W2.n_rows, W2.n_cols);
  eg2b1 = arma::zeros<arma::vec>(b1.n_rows);
}

void NeuralNetworkClassifier::take_ada_gradient_step() {
  eg2W1 += grad_W1 % grad_W1;
  W1 -= ada_alpha * (grad_W1 / arma::sqrt(eg2W1 + ada_eps));

  eg2b1 += grad_b1 % grad_b1;
  b1 -= ada_alpha * (grad_b1 / arma::sqrt(eg2b1 + ada_eps));

  eg2W2 += grad_W2 % grad_W2;
  W2 -= ada_alpha * (grad_W2 / arma::sqrt(eg2W2 + ada_eps));

  if (!fix_embeddings) {
    eg2E += grad_E % grad_E;
    E -= ada_alpha * (grad_E / arma::sqrt(eg2E + ada_eps));
  }
}

double NeuralNetworkClassifier::get_cost() { return loss; }

double NeuralNetworkClassifier::get_accuracy() { return accuracy; }

void NeuralNetworkClassifier::get_precomputed_features(
    std::vector<Sample>::const_iterator& begin,
    std::vector<Sample>::const_iterator& end,
    std::unordered_set<int>& retval) {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  for (std::vector<Sample>::const_iterator sample = begin; sample != end; ++ sample) {
    for (size_t i = 0; i < sample->attributes.size(); ++ i) {
      int fid = sample->attributes[i]* nr_feature_types + i;
      if (encoder.find(fid) != encoder.end()) { retval.insert(fid); }
    }
  }
  // TRACE_LOG("classifier: percentage of necessary precomputation: %lf%%",
  // (double)retval.size() / encoder.size() * 100);
}

void NeuralNetworkClassifier::precomputing() {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  std::unordered_set<int> features;
  for (std::unordered_map<int, size_t>::const_iterator rep = encoder.begin();
      rep != encoder.end(); ++ rep) { features.insert(rep->first); }
  precomputing(features);
}

void NeuralNetworkClassifier::precomputing(
    const std::unordered_set<int>& features) {
  saved.zeros();
  for (std::unordered_set<int>::const_iterator rep = features.begin();
      rep != features.end(); ++ rep) {
    int fid = (*rep);
    size_t rank = precomputation_id_encoder[fid];
    size_t aid = fid / nr_feature_types;
    size_t off = (fid % nr_feature_types)*embedding_size;
    saved.col(rank) =
      W1.submat(0, off, hidden_layer_size-1, off+embedding_size-1) * E.col(aid);
  }
  // TRACE_LOG("classifier: precomputed %d", features.size());
}

void NeuralNetworkClassifier::compute_gradient(
    std::vector<Sample>::const_iterator& begin,
    std::vector<Sample>::const_iterator& end,
    size_t batch_size) {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;

  grad_W1.zeros();
  grad_b1.zeros();
  grad_W2.zeros();
  grad_E.zeros();

  loss = 0; accuracy = 0;

  for (std::vector<Sample>::const_iterator sample = begin; sample != end; ++ sample) {
    const std::vector<int>& attributes = sample->attributes;
    const std::vector<double>& classes = sample->classes;

    arma::vec Y(classes);

    arma::uvec dropout_mask = arma::find(
        arma::randu<arma::vec>(hidden_layer_size) > dropout_probability );

    arma::vec hidden_layer = arma::zeros<arma::vec>(dropout_mask.n_rows);

    for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
      int aid = attributes[i];
      int fid = aid * nr_feature_types + i;
      std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
      if (rep != encoder.end()) {
        arma::uvec _;
        _ << rep->second;
        hidden_layer += saved.submat(dropout_mask, _);
      } else {
        arma::uvec __ = arma::linspace<arma::uvec>(off, off+embedding_size-1, embedding_size);
        hidden_layer += (W1.submat(dropout_mask, __)* E.col(aid));
      }
    }

    hidden_layer += b1(dropout_mask);
    // arma::vec cubic_hidden_layer = hidden_layer % hidden_layer % hidden_layer;
    arma::vec cubic_hidden_layer = arma::clamp(arma::pow(hidden_layer, 3), -50., 50);

    // Mat(nr_classes, hidden_layer_size) * Vec(hidden_layer_size)
    // arma::vec output = W2.cols(dropout_mask) * cubic_hidden_layer(dropout_mask);
    arma::vec output = W2.cols(dropout_mask) * cubic_hidden_layer;
    int opt_class = -1, correct_class = -1;
    for (size_t i = 0; i < nr_classes; ++ i) {
      if (classes[i] >= 0 && (opt_class < 0 || output(i) > output(opt_class))) {
        opt_class = i; }
      if (classes[i] == 1) { correct_class = i; }
    }

    arma::uvec classes_mask = arma::find(Y >= 0);
    double best = output(opt_class);
    output(classes_mask) = arma::exp(output(classes_mask) - best);
    double sum1 = output(correct_class);
    double sum2 = arma::sum(output(classes_mask));

    loss += (log(sum2) - log(sum1));
    if (classes[opt_class] == 1) { accuracy += 1; }

    // delta(classes_mask, 1)
    arma::vec delta =
      -(Y(classes_mask) - output(classes_mask) / sum2) / batch_size;

    grad_W2.submat(classes_mask, dropout_mask) += delta * cubic_hidden_layer.t();

    arma::vec grad_cubic_hidden_layer =
      W2.submat(classes_mask, dropout_mask).t() * delta;

    arma::vec grad_hidden_layer = 3 * grad_cubic_hidden_layer
                                  % hidden_layer
                                  % hidden_layer;

    grad_b1(dropout_mask) += grad_hidden_layer;

    for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
      int aid = attributes[i];
      int fid = aid * nr_feature_types + i;
      std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
      if (rep != encoder.end()) {
        arma::uvec _;
        _ << rep->second;
        grad_saved.submat(dropout_mask, _) += grad_hidden_layer;
      } else {
        arma::uvec __ = arma::linspace<arma::uvec>(off, off+embedding_size-1, embedding_size);
        grad_W1.submat(dropout_mask, __) += grad_hidden_layer * E.col(aid).t();
        if (!fix_embeddings) {
          grad_E.col(aid) += W1.submat(dropout_mask, __).t() * grad_hidden_layer;
        }
      }
    }
  }

  loss /= batch_size;
  accuracy /= batch_size;
}

void NeuralNetworkClassifier::compute_saved_gradient(
    const std::unordered_set<int>& features) {
  std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  for (std::unordered_set<int>::const_iterator rep = features.begin();
      rep != features.end(); ++ rep) {
    int fid = (*rep);
    size_t rank = encoder[fid];
    size_t aid = fid / nr_feature_types;
    size_t off = (fid % nr_feature_types)*embedding_size;

    grad_W1.submat(0, off, hidden_layer_size-1, off+embedding_size-1) +=
      grad_saved.col(rank) * E.col(aid).t();

    if (!fix_embeddings) {
      grad_E.col(aid) += 
        W1.submat(0, off, hidden_layer_size-1, off+ embedding_size-1).t()
        * grad_saved.col(rank);
    }
  }
}

void NeuralNetworkClassifier::add_l2_regularization() {
  loss += (lambda * .5 * (arma::dot(W1, W1)+ arma::dot(b1, b1)+ arma::dot(W2, W2)));
  if (!fix_embeddings) { loss += (lambda * .5 * arma::dot(E, E)); }

  grad_W1 += lambda * W1;
  grad_b1 += lambda * b1;
  grad_W2 += lambda * W2;
  if (!fix_embeddings) { grad_E += lambda * E; }
}

void NeuralNetworkClassifier::save(std::ofstream& ofs) {
#if 0
  E.save(ofs);
  W1.save(ofs);
  b1.save(ofs);
  W2.save(ofs);
  saved.save(ofs);
  boost::archive::text_oarchive oa(ofs);
  oa << precomputation_id_encoder;
#endif
}

void NeuralNetworkClassifier::load(std::ifstream& ifs) {
#if 0
  E.load(ifs);
  W1.load(ifs);
  b1.load(ifs);
  W2.load(ifs);
  saved.load(ifs);
  boost::archive::text_iarchive ia(ifs);
  ia >> precomputation_id_encoder;
  hidden_layer_size = b1.n_rows;
  nr_feature_types = W1.n_cols / E.n_rows;
  nr_classes = W2.n_rows;
  embedding_size = E.n_rows;
  info();
#endif
}

void NeuralNetworkClassifier::info() {
  TRACE_LOG("classifier: E(%d,%d)", E.n_rows, E.n_cols);
  TRACE_LOG("classifier: W1(%d,%d)", W1.n_rows, W1.n_cols);
  TRACE_LOG("classifier: b1(%d)", b1.n_rows);
  TRACE_LOG("classifier: W2(%d,%d)", W2.n_rows, W2.n_cols);
  TRACE_LOG("classifier: saved(%d,%d)", saved.n_rows, saved.n_cols);
  TRACE_LOG("classifier: precomputed size=%d", precomputation_id_encoder.size());
  TRACE_LOG("classifier: hidden layer size=%d", hidden_layer_size);
  TRACE_LOG("classifier: embedding size=%d", embedding_size);
  TRACE_LOG("classifier: number of classes=%d", nr_classes);
  TRACE_LOG("classifier: number of feature types=%d", nr_feature_types);
}

} //  namespace depparser
} //  namespace ltp
