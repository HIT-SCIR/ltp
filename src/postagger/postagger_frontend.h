#ifndef __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__
#define __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__

#include "utils/cfgparser.hpp"
#include "framework/frontend.h"
#include "postagger/options.h"
#include "postagger/postagger.h"
#include "postagger/decoder.h"
#include "postagger/constrainutil.hpp"

namespace ltp {
namespace postagger {

class PostaggerFrontend: public Postagger, framework::Frontend {
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

  /*
   * build postags dictionary
   */
  void build_configuration(void);

  /*
   * build the feature space
   */
  void build_feature_space(void);

  /*
   * the evaluating process
   */
  void evaluate(double &p);

  /**
   *
   */
  void cleanup_decode_context();

  /*
   * do feature trauncation on the model. create a model duplation
   * on the model and return their
   *
   *  @param[in]  feature_group_updated_times   the updated time of feature
   *                                            group
   *  @return     Model                         the duplication of the model
   */
  Model * erase_rare_features(int * feature_group_updated_times = 0);
protected:
  /*
   * extract feature from the instance. If create handler is configured,
   *
   *  @param[in]  inst    the instance
   *  @param[in]  create    use to specify create process
   */
  void extract_features(const Instance* inst, bool create);

  /*
   * cache all the score for the certain instance.
   *
   *  @param[in/out]  inst  the instance
   *  @param[in]    use_avg use to specify use average parameter
   */
  void calculate_scores(const Instance* inst, bool use_avg);

  /*
   * collect feature when given the tags index
   *
   *  @param[in]    uni_features  the unigram features
   *  @param[in]    tagsidx the tags index
   *  @param[out]   vec   the output sparse vector
   */
  void collect_features(const math::Mat< math::FeatureVector* > & uni_features,
      const std::vector<int> & tagsidx,
      math::SparseVec & vec);

  /*
   * decode the group information for feature represented in sparse vector,
   * increase their updated time
   *
   *  @param[in]  vec           the feature vector
   *  @param[out] updated_time  the updated time
   */
  void increase_group_updated_time(const ltp::math::SparseVec & vec,
                                   int * feature_group_updated_time);
private:
  ModelOptions model_opt;
  TrainOptions train_opt;
  TestOptions  test_opt;
  DumpOptions  dump_opt;
protected:
  std::vector< Instance * > train_dat;  //! The training data.
  Decoder * decoder;                    //! The decoder.
  DecodeContext* decode_context;        //! The decode context
  ScoreMatrix* score_matrix;            //! The score matrix
};

}
}

#endif  //  end for __LTP_POSTAGGER_POSTAGGER_FRONTEND_H__
