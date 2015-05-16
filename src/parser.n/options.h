#ifndef __LTP_PARSERN_OPTIONS_H__
#define __LTP_PARSERN_OPTIONS_H__

#include <iostream>

namespace ltp {
namespace depparser {

struct SpecialOption {
  static std::string UNKNOWN;
  static std::string NIL;
  static std::string ROOT;
};

struct BasicOption {
  std::string model_file;     //! The path to the model.
  std::string root;           //! The root.
};

struct AdaOption {
  double ada_eps;             //! Eps used in AdaGrad
  double ada_alpha;           //! Alpha used in AdaGrad
  double lambda;              //! TODO not known.
  double dropout_probability; //! The probability for dropout.
};

struct NetworkOption {
  int hidden_layer_size;    //! Size for hidden layer.
  int embedding_size;       //! Size for embedding.
};

struct LearnOption:
  public BasicOption,
  public AdaOption,
  public NetworkOption {
  std::string reference_file;   //! The path to the reference file.
  std::string devel_file;       //! The path to the devel file.
  std::string embedding_file;   //! The path to the embedding.
  std::string cluster_file;     //! The path to the cluster file, actived in use-cluster.
  std::string oracle;           //! The oracle type, can be [static, nondet, explore]
  int word_cutoff;              //! The frequency of rare word, word lower than that
                                //! will be cut off.
  int max_iter;                 //! The maximum iteration.
  double init_range;            //!
  int batch_size;               //! The Size of batch.
  int nr_precomputed;           //! The number of precomputed features
  int evaluation_stops;         //!
  int clear_gradient_per_iter;  //! clear gradient each iteration.
  bool save_intermediate;       //! Save model whenever see an improved UAS.
  bool fix_embeddings;          //! Not tune the embedding when learning the parameters
  bool use_distance;            //! Specify to use distance feature.
  bool use_valency;             //! Specify to use valency feature.
  bool use_cluster;             //! Specify to use cluster feature.
};

struct TestOption:
  public BasicOption {
  std::string input_file;   //! The path to the input file.
  std::string output_file;  //! The path to the output file.
  bool evaluate;
};

} //  namespace depparser
} //  namespace ltp

#endif  //  end for __LTP_PARSERN_OPTIONS_H__
