#ifndef __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__
#define __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__

#include "framework/frontend.h"
#include "postagger/options.h"
#include "postagger/postagger.h"
#include "postagger/decoder.h"
// #include "postagger/constrainutil.hpp"

namespace ltp {
namespace postagger {

using framework::Frontend;
using framework::ViterbiDecoder;
using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;

class PostaggerFrontend: public Postagger, framework::Frontend {
private:
  ViterbiDecoder* decoder;            //! The decoder.
  ViterbiFeatureContext* ctx;         //! The decode context
  ViterbiScoreMatrix* scm;            //! The score matrix
  std::vector<Instance *> train_dat;  //! The training data.

  TrainOptions train_opt;
  TestOptions  test_opt;
  DumpOptions  dump_opt;

  static const std::string model_header;
public:
  PostaggerFrontend(const std::string& reference_file,
      const std::string& holdout_file,
      const std::string& model_name,
      const std::string& algorithm,
      const int max_iter,
      const int rare_feature_threshold);

  PostaggerFrontend(const std::string& model_file,
      const std::string& input_file,
      const std::string& lexicon_file,
      bool evaluate);

  PostaggerFrontend(const std::string& model_file);

  ~PostaggerFrontend();

  void train(void); //  Training
  void test(void);  //  Testing
  void dump(void);  //  Dumping
private:

  /**
   * read instance from the file, and store the instances
   * in train data.
   *
   *  @param[in]  file_name   the file name
   */
  bool read_instances(const char* file_name);

  /**
   * build postags dictionary
   */
  void build_configuration(void);

  /**
   * build the feature space
   */
  void build_feature_space(void);

  /**
   * the evaluating process
   */
  void evaluate(double &p);

protected:
  /*
   * collect feature when given the tags index
   *
   *  @param[in]    uni_features  the unigram features
   *  @param[in]    tagsidx the tags index
   *  @param[out]   vec   the output sparse vector
   */
  void collect_features(const math::Mat<math::FeatureVector *> & uni_features,
      const std::vector<int> & tagsidx,
      math::SparseVec& vec);
};

}
}

#endif  //  end for __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__
