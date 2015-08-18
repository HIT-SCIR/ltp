#ifndef __LTP_NER_NER_FRONTEND_H__
#define __LTP_NER_NER_FRONTEND_H__

#include "framework/frontend.h"
#include "framework/decoder.h"
#include "ner/options.h"
#include "ner/ner.h"
#include "utils/unordered_set.hpp"
#include "ner/decoder.h"

namespace ltp {
namespace ner {

using framework::Frontend;
using framework::ViterbiDecoder;
using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;

class NamedEntityRecognizerFrontend: public NamedEntityRecognizer, Frontend {
private:
  NERViterbiDecoderWithMarginal decoder;
  ViterbiFeatureContext ctx;
  ViterbiScoreMatrix scm;
  std::vector<Instance *> train_dat;

  TrainOptions train_opt;
  TestOptions test_opt;
  DumpOptions dump_opt;

public:
  //! The learning model constructor.
  NamedEntityRecognizerFrontend(const std::string& reference_file,
      const std::string& holdout_file,
      const std::string& model_file,
      const std::string& algorithm,
      const int max_iter,
      const int rare_feature_threshold);

  //! The testing model constructor.
  NamedEntityRecognizerFrontend(const std::string& model_file,
      const std::string& input_file,
      bool evaluate,
      bool sequence_prob = false,
      bool marginal_prob = false);

  //! The dumping model constructor.
  NamedEntityRecognizerFrontend(const std::string& model_file);

  ~NamedEntityRecognizerFrontend();

  void train(void); //! Training
  void test(void);  //! Testing
  void dump(void);  //! Dumping

private:
  typedef std::unordered_set<std::string> set_t;
  /**
   * read instances from file and store them in train_dat
   *
   *  @param[in]  file_name   the filename
   *  @return     bool        true on success, otherwise false
   */
  bool read_instance(const std::string& file_name);
  void build_configuration(void);
  void build_feature_space(void);
  /**
   * collect feature when given the tags index
   *
   *  @param[in]    uni_features  the unigram features
   *  @param[in]    tagsidx the tags index
   *  @param[out]   vec   the output sparse vector
   */
  void collect_features(const math::Mat< math::FeatureVector* > & uni_features,
      const std::vector<int> & tagsidx,
      math::SparseVec & vec);

  void evaluate(double& f_score);
};

} //  namespace ner
} //  namespace ltp

#endif  //  end for __LTP_NER_NER_FRONTEND_H__
