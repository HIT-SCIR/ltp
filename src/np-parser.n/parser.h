#ifndef __LTP_PARSERN_PARSER_H__
#define __LTP_PARSERN_PARSER_H__

#include "utils/smartmap.hpp"
#include "utils/unordered_map.hpp"
#include "parser.n/system.h"
#include "parser.n/context.h"
#include "parser.n/classifier.h"
#include "framework/serializable.h"
#include "Eigen/Dense"

namespace ltp {
namespace depparser {

class NeuralNetworkParser: public framework::Serializable {
protected:
  /*arma::mat W1;
  arma::mat W2;
  arma::mat E;
  arma::vec b1;
  arma::mat saved;*/
  Eigen::MatrixXd W1;
  Eigen::MatrixXd W2;
  Eigen::MatrixXd E;
  Eigen::VectorXd b1;
  Eigen::MatrixXd saved;

  utility::IndexableSmartMap forms_alphabet;
  utility::IndexableSmartMap postags_alphabet;
  utility::IndexableSmartMap deprels_alphabet;
  utility::IndexableSmartMap cluster4_types_alphabet;
  utility::IndexableSmartMap cluster6_types_alphabet;
  utility::IndexableSmartMap cluster_types_alphabet;

  std::unordered_map<int, size_t> precomputation_id_encoder;
  std::unordered_map<int, int> form_to_cluster4;
  std::unordered_map<int, int> form_to_cluster6;
  std::unordered_map<int, int> form_to_cluster;

  NeuralNetworkClassifier classifier;
  TransitionSystem system;
  std::string root;

  size_t kNilForm;
  size_t kNilPostag;
  size_t kNilDeprel;
  size_t kNilDistance;
  size_t kNilValency;
  size_t kNilCluster4;
  size_t kNilCluster6;
  size_t kNilCluster;

  size_t kFormInFeaturespace;
  size_t kPostagInFeaturespace;
  size_t kDeprelInFeaturespace;
  size_t kDistanceInFeaturespace;
  size_t kValencyInFeaturespace;
  size_t kCluster4InFeaturespace;
  size_t kCluster6InFeaturespace;
  size_t kClusterInFeaturespace;
  size_t kFeatureSpaceEnd;

  size_t nr_feature_types;

  bool use_distance;
  bool use_valency;
  bool use_cluster;

  static const std::string model_header;
public:
  NeuralNetworkParser();

  void get_features(const State& state,
      std::vector<int>& features);

  void get_features(const State& state,
      const std::vector<int>& cluster4,
      const std::vector<int>& cluster6,
      const std::vector<int>& cluster,
      std::vector<int>& features);

  void predict(const Instance& inst, std::vector<int>& heads,
      std::vector<std::string>& deprels);

  bool load(const std::string& filename);

  void save(const std::string& filename);

protected:
  void get_context(const State& state, Context* ctx);

  void get_basic_features(const Context& ctx,
      const std::vector<int>& forms,
      const std::vector<int>& postags,
      const std::vector<int>& deprels,
      std::vector<int>& features);

  void get_distance_features(const Context& ctx,
      std::vector<int>& features);

  void get_valency_features(const Context& ctx,
      const std::vector<int>& nr_left_children,
      const std::vector<int>& nr_right_children,
      std::vector<int>& features);

  void get_cluster_features(const Context& ctx,
      const std::vector<int>& cluster4,
      const std::vector<int>& cluster6,
      const std::vector<int>& cluster,
      std::vector<int>& features);

  void build_feature_space();
  void setup_system();
  void report();

  void transduce_instance_to_dependency(const Instance& data,
      Dependency* dependency, bool with_dependencies);

  void get_cluster_from_dependency(const Dependency& dependency,
      std::vector<int>& cluster4,
      std::vector<int>& cluster6,
      std::vector<int>& cluster);

  template<class Matrix> void write_matrix(std::ostream& os,
      const Matrix& matrix);
  template<class Matrix> void read_matrix(std::istream& is,
      Matrix& matrix);
  template<class Vector> void write_vector(std::ostream& os,
      const Vector& vector);
  template<class Vector> void read_vector(std::istream& is,
      Vector& vector);
};

} //  namespace depparser
} //  namespace ltp

#endif  //  end for __LTP_PARSERN_PARSER_H__
