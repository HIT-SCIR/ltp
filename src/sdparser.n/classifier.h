#ifndef __LTP_PARSERN_CLASSIFIER_H__
#define __LTP_PARSERN_CLASSIFIER_H__

#include <iostream>
#include <vector>
#include "parser.n/options.h"
#include "utils/unordered_map.hpp"
#include "utils/unordered_set.hpp"
#include "Eigen/Dense"

namespace ltp {
namespace depparser {

struct Sample {
  std::vector<int> attributes;  //! sparse vector of attributes
  std::vector<double> classes;  //! dense vector of classes

  Sample();
  Sample(const std::vector<int>& attributes,
      const std::vector<double>& classes);
};

class NeuralNetworkClassifier {
private:
  // The weight group.
  /*arma::mat& W1;         // Mat: hidden_layer_size X (nr_feature_types * embedding_size)
  arma::mat& W2;         // Mat: nr_classes X hidden_layer_size
  arma::mat& E;          // Mat: nr_objects X embedding_size
  arma::vec& b1;         // Vec: hidden_layer_size

  arma::mat grad_W1;
  arma::vec grad_b1;
  arma::mat grad_W2;
  arma::mat grad_E;

  arma::mat eg2W1;
  arma::mat eg2W2;
  arma::mat eg2E;
  arma::vec eg2b1;*/

  Eigen::MatrixXd& W1;
  Eigen::MatrixXd& W2;
  Eigen::MatrixXd& E;
  Eigen::VectorXd& b1;
  Eigen::MatrixXd& saved;

  Eigen::MatrixXd grad_W1;
  Eigen::MatrixXd grad_W2;
  Eigen::MatrixXd grad_E;
  Eigen::VectorXd grad_b1;
  Eigen::MatrixXd grad_saved;

  Eigen::MatrixXd eg2W1;
  Eigen::MatrixXd eg2W2;
  Eigen::MatrixXd eg2E;
  Eigen::VectorXd eg2b1;

  double loss;
  double accuracy;

  // Precomputed matrix
  /*arma::mat& saved;      // Mat: encoder.size() X hidden_layer_size
  arma::mat grad_saved; // Mat: encoder.size() X hidden_layer_size*/

private:
  // The configuration
  size_t embedding_size;      //! The size of the embedding.
  size_t hidden_layer_size;   //! The size of the hidden layer
  size_t nr_objects;          //! The sum of forms, postags and deprels
  size_t nr_feature_types;    //! The number of feature types
  size_t nr_classes;          //! The number of classes

  size_t batch_size;
  size_t nr_threads;
  bool fix_embeddings;

  double dropout_probability;
  double lambda;
  double ada_eps;
  double ada_alpha;

  std::unordered_map<int, size_t>& precomputation_id_encoder;

  bool initialized;
public:
  /*NeuralNetworkClassifier(arma::mat& W1, arma::mat& W2, arma::mat& E,
      arma::vec& b1, arma::mat& saved,
      std::unordered_map<int, size_t>& precomputation_id_encoder);*/

  NeuralNetworkClassifier(
      Eigen::MatrixXd& _W1,
      Eigen::MatrixXd& _W2,
      Eigen::MatrixXd& _E,
      Eigen::VectorXd& _b1,
      Eigen::MatrixXd& _saved,
      std::unordered_map<int, size_t>& encoder);

  /**
   * Initialize the neural network
   *
   *  @param[in]  nr_forms        The size of vocabulary.
   *  @param[in]  nr_postags      The size of postags.
   *  @param[in]  nr_labels       The size of label set.
   *  @param[in]  nr_tokens       (?)
   *  @param[in]  embedding_size  The size of embeddings
   *  @param[in]  hidden_size     The size of hidden layer
   */
  void initialize(int nr_objects,
      int nr_classes,
      int nr_feature_types,
      const LearnOption& opt,
      const std::vector< std::vector<double> >& embeddings,
      const std::vector<int>& precomputed_features
      );

  /**
   * Calculate scores for the given features.
   *
   *  @param[in]  features  The features
   *  @param[out] retval    The calculated score
   */
  void score(const std::vector<int>& attributes, std::vector<double>& retval);

  void canonical();
  void info();

  //!
  void compute_ada_gradient_step(
      std::vector<Sample>::const_iterator begin,
      std::vector<Sample>::const_iterator end
      );

  //!
  void initialize_gradient_histories();

  void take_ada_gradient_step();

  //!
  double get_cost();

  //!
  double get_accuracy();

  /**
   * Collect the indices from samples and put them into a set.
   *
   *  @param[in]  samples The samples
   *  @param[out] retval  The set of indicies
   */
  void get_precomputed_features(std::vector<Sample>::const_iterator& begin,
      std::vector<Sample>::const_iterator& end,
      std::unordered_set<int>& retval);

  void precomputing();

  void precomputing(const std::unordered_set<int>& candidates);

  void compute_gradient(std::vector<Sample>::const_iterator& begin,
      std::vector<Sample>::const_iterator& end,
      size_t batch_size);

  void compute_saved_gradient(
      const std::unordered_set<int>& precomputed_indices);

  void add_l2_regularization();
};

} //  namespace depparser
} //  namespace ltp

#endif  //  end for __LTP_PARSERN_CLASSIFIER_H__
