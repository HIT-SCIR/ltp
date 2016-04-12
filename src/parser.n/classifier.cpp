#include "parser.n/classifier.h"
#include "utils/logging.hpp"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace ltp {
namespace depparser {

Sample::Sample() {}

Sample::Sample(const std::vector<int>& _attributes,
    const std::vector<double>& _classes)
  : attributes(_attributes), classes(_classes) {
}


NeuralNetworkClassifier::NeuralNetworkClassifier(
    Eigen::MatrixXd& _W1,
    Eigen::MatrixXd& _W2,
    Eigen::MatrixXd& _E,
    Eigen::VectorXd& _b1,
    Eigen::MatrixXd& _saved,
    std::unordered_map<int, size_t>& encoder)
  : initialized(false), W1(_W1), W2(_W2), E(_E), b1(_b1),
  saved(_saved), precomputation_id_encoder(encoder),
  embedding_size(0), hidden_layer_size(0),
  nr_objects(0), nr_feature_types(0), nr_classes(0) {}

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
  W1 = Eigen::MatrixXd::Random(nrows, ncols) * sqrt(6. / (nrows+ ncols));
  b1 = Eigen::VectorXd::Random(nrows) * sqrt(6. / (nrows+ ncols));

  nrows = _nr_classes;  //
  ncols = hidden_layer_size;
  W2 = Eigen::MatrixXd::Random(nrows, ncols)* sqrt(6./ (nrows+ ncols));

  // Initialized the embedding
  nrows = embedding_size;
  ncols= _nr_objects;

  E = Eigen::MatrixXd::Random(nrows, ncols)* opt.init_range;

  for (size_t i = 0; i < embeddings.size(); ++ i) {
    const std::vector<double>& embedding = embeddings[i];
    int id = embedding[0];
    for (unsigned j = 1; j < embedding.size(); ++ j) {
      E(j-1, id) = embedding[j];
    }
  }

  grad_W1 = Eigen::MatrixXd::Zero(W1.rows(), W1.cols());
  grad_b1 = Eigen::VectorXd::Zero(b1.rows());
  grad_W2 = Eigen::MatrixXd::Zero(W2.rows(), W2.cols());
  grad_E = Eigen::MatrixXd::Zero(E.rows(), E.cols());

  // Initialized the precomputed features
  std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  size_t rank = 0;

  for (size_t i = 0; i < precomputed_features.size(); ++ i) {
    int fid = precomputed_features[i];
    encoder[fid] = rank ++;
  }

  saved = Eigen::MatrixXd::Zero(hidden_layer_size, encoder.size());
  grad_saved = Eigen::MatrixXd::Zero(hidden_layer_size, encoder.size());

  //
  initialize_gradient_histories();

  initialized = true;

  info();
  INFO_LOG("classifier: size of batch = %d", batch_size);
  INFO_LOG("classifier: alpha = %e", ada_alpha);
  INFO_LOG("classifier: eps = %e", ada_eps);
  INFO_LOG("classifier: lambda = %e", lambda);
  INFO_LOG("classifier: fix embedding = %s", (fix_embeddings? "true": "false"));
}

void NeuralNetworkClassifier::score(const std::vector<int>& attributes,
    std::vector<double>& retval) {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;
  // arma::vec hidden_layer = arma::zeros<arma::vec>(hidden_layer_size);
  Eigen::VectorXd hidden_layer = Eigen::VectorXd::Zero(hidden_layer_size);

  for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
    int aid = attributes[i];
    int fid = aid * nr_feature_types + i;
    std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
    if (rep != encoder.end()) {
      /* hidden_layer += saved.col(rep->second);*/
      hidden_layer += saved.col(rep->second);
    } else {
      // W1[0:hidden_layer, off:off+embedding_size] * E[fid:]'
      hidden_layer += W1.block(0, off, hidden_layer_size, embedding_size)* E.col(aid);
    }
  }

  hidden_layer += b1;

  Eigen::VectorXd output = W2 * Eigen::VectorXd(hidden_layer.array().cube());
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
  /*grad_saved.zeros();*/
  grad_saved.setZero();
  compute_gradient(begin, end, end- begin);
  compute_saved_gradient(precomputed_features);

  // add regularizer.
  add_l2_regularization();
}

void NeuralNetworkClassifier::initialize_gradient_histories() {
  eg2W1 = Eigen::MatrixXd::Zero(W1.rows(), W1.cols());
  eg2b1 = Eigen::VectorXd::Zero(b1.rows());
  eg2W2 = Eigen::MatrixXd::Zero(W2.rows(), W2.cols());
  eg2E = Eigen::MatrixXd::Zero(E.rows(), E.cols());
}

void NeuralNetworkClassifier::take_ada_gradient_step() {
  eg2W1 += Eigen::MatrixXd(grad_W1.array().square());
  W1 -= ada_alpha * Eigen::MatrixXd(grad_W1.array() / (eg2W1.array() + ada_eps).sqrt());

  eg2b1 += Eigen::VectorXd(grad_b1.array().square());
  b1 -= ada_alpha * Eigen::VectorXd(grad_b1.array() / (eg2b1.array() + ada_eps).sqrt());

  eg2W2 += Eigen::MatrixXd(grad_W2.array().square());
  W2 -= ada_alpha * Eigen::MatrixXd(grad_W2.array() / (eg2W2.array() + ada_eps).sqrt());

  if (!fix_embeddings) {
    eg2E += Eigen::MatrixXd(grad_E.array().square());
    E -= ada_alpha * Eigen::MatrixXd(grad_E.array() / (eg2E.array() + ada_eps).sqrt());
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
  // INFO_LOG("classifier: percentage of necessary precomputation: %lf%%",
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
  saved.setZero();
  for (std::unordered_set<int>::const_iterator rep = features.begin();
      rep != features.end(); ++ rep) {
    int fid = (*rep);
    size_t rank = precomputation_id_encoder[fid];
    size_t aid = fid / nr_feature_types;
    size_t off = (fid % nr_feature_types)*embedding_size;
    saved.col(rank) = W1.block(0, off, hidden_layer_size, embedding_size)* E.col(aid);
  }
  // INFO_LOG("classifier: precomputed %d", features.size());
}

void NeuralNetworkClassifier::compute_gradient(
    std::vector<Sample>::const_iterator& begin,
    std::vector<Sample>::const_iterator& end,
    size_t batch_size) {
  const std::unordered_map<int, size_t>& encoder = precomputation_id_encoder;

  grad_W1.setZero();
  grad_b1.setZero();
  grad_W2.setZero();
  grad_E.setZero();

  loss = 0; accuracy = 0;

  // special for Eigen::XXX::Random
  double mask_prob = dropout_probability* 2- 1;
  for (std::vector<Sample>::const_iterator sample = begin; sample != end; ++ sample) {
    const std::vector<int>& attributes = sample->attributes;
    const std::vector<double>& classes = sample->classes;

    Eigen::VectorXd Y = Eigen::VectorXd::Map(&classes[0], classes.size());
    Eigen::VectorXd _ = (Eigen::ArrayXd::Random(hidden_layer_size) > mask_prob).select(
        Eigen::VectorXd::Ones(hidden_layer_size),
        Eigen::VectorXd::Zero(hidden_layer_size));
    Eigen::VectorXd hidden_layer = Eigen::VectorXd::Zero(hidden_layer_size);

    for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
      int aid = attributes[i];
      int fid = aid * nr_feature_types + i;
      std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
      if (rep != encoder.end()) {
        hidden_layer += _.asDiagonal() * saved.col(rep->second);
      } else {
        hidden_layer +=
          _.asDiagonal() * W1.block(0, off, hidden_layer_size, embedding_size) * E.col(aid);
      }
    }

    hidden_layer += _.asDiagonal() * b1;

    Eigen::VectorXd cubic_hidden_layer = hidden_layer.array().cube().min(50).max(-50);
    Eigen::VectorXd output = W2 * cubic_hidden_layer;

    int opt_class = -1, correct_class = -1;
    for (size_t i = 0; i < nr_classes; ++ i) {
      if (classes[i] >= 0 && (opt_class < 0 || output(i) > output(opt_class))) {
        opt_class = i; }
      if (classes[i] == 1) { correct_class = i; }
    }

    /*arma::uvec classes_mask = arma::find(Y >= 0);*/
    Eigen::VectorXd __ = (Y.array() >= 0).select(
        Eigen::VectorXd::Ones(nr_classes),
        Eigen::VectorXd::Zero(nr_classes));
    double best = output(opt_class);
    output = __.asDiagonal() * Eigen::VectorXd((output.array() - best).exp());
    double sum1 = output(correct_class);
    double sum2 = output.sum();

    loss += (log(sum2) - log(sum1));
    if (classes[opt_class] == 1) { accuracy += 1; }

    Eigen::VectorXd delta =
      -(__.asDiagonal()*Y - Eigen::VectorXd(output.array()/sum2)) / batch_size;

    grad_W2 += delta * cubic_hidden_layer.transpose();
    Eigen::VectorXd grad_cubic_hidden_layer = _.asDiagonal() * W2.transpose() * delta;

    Eigen::VectorXd grad_hidden_layer =
      3 * grad_cubic_hidden_layer.array() * hidden_layer.array().square();

    grad_b1 += grad_hidden_layer;

    for (size_t i = 0, off = 0; i < attributes.size(); ++ i, off += embedding_size) {
      int aid = attributes[i];
      int fid = aid * nr_feature_types + i;
      std::unordered_map<int, size_t>::const_iterator rep = encoder.find(fid);
      if (rep != encoder.end()) {
        grad_saved.col(rep->second) += grad_hidden_layer;
      } else {
        grad_W1.block(0, off, hidden_layer_size, embedding_size) +=
          grad_hidden_layer * E.col(aid).transpose();
        if (!fix_embeddings) {
          grad_E.col(aid) +=
            W1.block(0, off, hidden_layer_size, embedding_size).transpose()* grad_hidden_layer;
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

    grad_W1.block(0, off, hidden_layer_size, embedding_size) +=
      grad_saved.col(rank) * E.col(aid).transpose();

    if (!fix_embeddings) {
      grad_E.col(aid) +=
        W1.block(0, off, hidden_layer_size, embedding_size).transpose() * grad_saved.col(rank);
    }
  }
}

void NeuralNetworkClassifier::add_l2_regularization() {
  loss += lambda * .5 * (W1.squaredNorm() + b1.squaredNorm() + W2.squaredNorm());
  if (!fix_embeddings) {
    loss += lambda * .5 * E.squaredNorm();
  }

  grad_W1 += lambda * W1;
  grad_b1 += lambda * b1;
  grad_W2 += lambda * W2;
  if (!fix_embeddings) { grad_E += lambda * E; }
}

void NeuralNetworkClassifier::info() {
  INFO_LOG("classifier: E(%d,%d)", E.rows(), E.cols());
  INFO_LOG("classifier: W1(%d,%d)", W1.rows(), W1.cols());
  INFO_LOG("classifier: b1(%d)", b1.rows());
  INFO_LOG("classifier: W2(%d,%d)", W2.rows(), W2.cols());
  INFO_LOG("classifier: saved(%d,%d)", saved.rows(), saved.cols());
  INFO_LOG("classifier: precomputed size=%d", precomputation_id_encoder.size());
  INFO_LOG("classifier: hidden layer size=%d", hidden_layer_size);
  INFO_LOG("classifier: embedding size=%d", embedding_size);
  INFO_LOG("classifier: number of classes=%d", nr_classes);
  INFO_LOG("classifier: number of feature types=%d", nr_feature_types);
}

void NeuralNetworkClassifier::canonical() {
  hidden_layer_size = b1.rows();
  nr_feature_types = W1.cols() / E.rows();
  nr_classes = W2.rows();
  embedding_size = E.rows();
}

} //  namespace depparser
} //  namespace ltp
