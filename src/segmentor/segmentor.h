#ifndef __LTP_SEGMENTOR_SEGMENTOR_H__
#define __LTP_SEGMENTOR_SEGMENTOR_H__

#include "utils/cfgparser.hpp"
#include "segmentor/model.h"
#include "segmentor/options.h"
#include "segmentor/decoder.h"
#include "segmentor/rulebase.h"

namespace ltp {
namespace segmentor {

class Segmentor {
public:
  Segmentor();
  Segmentor(ltp::utility::ConfigParser & cfg);
  ~Segmentor();

  /**
   * The only publich method in the Segmentor class. Parameters are passed to
   * the segmentor by a ::utility::ConfigParser class.
   */
  void run();

protected:
  /**
   * Parse the configuration
   *
   *  @param[in]  cfg         the config class
   *  @return     bool        return true on success, otherwise false
   */
  virtual bool parse_cfg(ltp::utility::ConfigParser & cfg);


  /**
   * Read instances from file and store them in train_dat
   *
   *  @param[in]  file_name   the filename
   *  @return     bool        true on success, otherwise false
   */
  bool read_instance( const char * file_name );


  /**
   * Build configuration before model training. Three things are done
   * during the building configuration pharse:
   *
   *  1. Build tag sets;
   *  2. Collect internal word map;
   *  3. Record word frequency.
   */
  virtual void build_configuration(void);


  /**
   * Build feature space.
   */
  virtual void build_feature_space(void);

  /**
   * Perform setup preparation for the training phase.
   */
  virtual bool train_setup(void);

  /**
   * Perform the passive aggressive training.
   *
   *  @param[in]  nr_errors The number of errors.
   */
  virtual void train_passive_aggressive(int nr_errors);

  /**
   * Perform the averaged perceptron training.
   */
  virtual void train_averaged_perceptron();

  /**
   * Return the flush time after each iteration.
   */
  virtual int get_timestamp(void);

  /**
   *
   */
  virtual void set_timestamp(int ts);

  /**
   * The main training process, the training scheme can be summarized as
   *
   *  1. Building configuration
   *  2. Building feature space
   *  3. Building updated time counter
   *  4. Iterate over the 
   */
  virtual void train(void);


  /*
   * The main evaluating process.
   *
   *  @param[out]   p   The precise
   *  @param[out]   r   The recall
   *  @param[out]   f   The F-score
   */
  virtual void evaluate(double &p, double &r, double &f);

  /**
   * Perform setup preparation for the training phase.
   */
  virtual bool test_setup(void);

  /**
   * The main testing process
   */
  void test(void);

  /**
   * The dumping model process
   */
  void dump(void);


  /**
   * Extract features from one instance,
   *
   *  @param[in/out]  inst    The instance
   *  @param[in]      create  If create is true, create feature for new
   *                          feature, otherwise not create.
   */
  virtual void extract_features(Instance * inst, bool create = false);

  /**
   * build words from tags for certain instance
   *
   *  @param[in/out]  inst      the instance
   *  @param[in]      tagsidx   the index of tags
   *  @param[out]     words     the output words
   *  @param[in]      begtag0   first of the word begin tag
   *  @param[in]      begtag1   second of the word begin tag
   */
  void build_words(Instance * inst,
                   const std::vector<int> & tagsidx,
                   std::vector<std::string> & words,
                   int beg_tag0,
                   int beg_tag1 = -1);

  /**
   * cache all the score for the certain instance.
   *
   *  @param[in/out]  inst      the instance
   *  @param[in]      use_avg   use to specify use average parameter
   */
  virtual void calculate_scores(Instance * inst, bool use_avg);


  /**
   * collect feature when given the tags index
   *
   *  @param[in]    inst    the instance
   *  @param[in]    tagsidx the tags index
   *  @param[out]   vec     the output sparse vector
   */
  void collect_features(const math::Mat< math::FeatureVector* >& uni_features,
                        Model* model,
                        Instance* inst,
                        const std::vector<int> & tagsidx,
                        ltp::math::SparseVec & vec);

  /**
   *
   *
   *
   *
   */
  virtual void collect_correct_and_predicted_features(Instance * inst);

  /**
   * Decode the group information for feature represented in sparse vector,
   * increase their updated time
   *
   *  @param[in]  vec           the feature vector
   *  @param[out] updated_time  the updated time
   */
  void increase_group_updated_time(const ltp::math::SparseVec & vec,
                                   int * updated_time);

  /**
   * do feature selection by erasing the rare feature. create a new model
   * without rare feature (only witness a few times) according the original
   * model.
   *
   *  @param[in]  nr_updates  the number of update times
   *  @return     Model       the model without rare feature
   */
  virtual Model * erase_rare_features(const int * nr_updates = NULL);

  /**
   * Remove the unigram features, bigram features from
   */
  void cleanup_decode_context(void);

protected:
  bool  __TRAIN__;  /*< The training flag */
  bool  __TEST__;   /*< The testing flag */
  bool  __DUMP__;   /*< The dump flag */

  //!
  int timestamp;

  //! The training options.
  TrainOptions* train_opt;

  //! The testing options.
  TestOptions* test_opt;

  //! The dump options.
  DumpOptions* dump_opt;

  //! The model.
  Model * model;

  //! The pointer to the decoder;
  Decoder * decoder;

  //! The pointer to the basic_rule;
  rulebase::RuleBase * baseAll;

  //! The collection of the training data.
  std::vector< Instance * > train_dat;

  //! the gold features.
  math::SparseVec correct_features;
  //! the predicted features.
  math::SparseVec predicted_features;
  //!
  math::SparseVec updated_features;

  //! The feature cache.
  math::Mat< math::FeatureVector *> uni_features;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_H__
