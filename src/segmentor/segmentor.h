#ifndef __LTP_SEGMENTOR_SEGMENTOR_H__
#define __LTP_SEGMENTOR_SEGMENTOR_H__

#include "utils/cfgparser.hpp"
#include "segmentor/model.h"
#include "segmentor/options.h"
#include "segmentor/decoder.h"
#include "segmentor/decode_context.h"
#include "segmentor/score_matrix.h"
#include "segmentor/rulebase.h"

namespace ltp {
namespace segmentor {

class Segmentor {
public:
  Segmentor();
  Segmentor(ltp::utility::ConfigParser & cfg);
  ~Segmentor();

  /**
   * The only public method in the Segmentor class. Parameters are passed to
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

  virtual void cleanup_decode_context();


  /**
   * Perform the averaged perceptron training.
   */
  virtual void train_averaged_perceptron();


  /**
   * Return the flush time after each iteration.
   *
   *  @return int   The timestamp.
   */
  virtual int get_timestamp(void);


  /**
   * Set the timestamp to certain value.
   *
   *  @param[in]  ts  The timestamp.
   */
  virtual void set_timestamp(int ts);

  /**
   * The main training process, the training scheme can be summarized as
   *
   *  1. Building configuration
   *  2. Building feature space
   *  3. Building updated time counter
   *  4. Iterate over the training instances
   */
  virtual void train(void);


  /**
   * The main evaluating process.
   *
   *  @param[out]  p   The precise
   *  @param[out]  r   The recall
   *  @param[out]  f   The F-score
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
   * Build lexicon match state of the instance
   *
   *  @param[in/out]  inst    The instance.
   *
   */
  virtual void build_lexicon_match_state(Instance* inst);


  /**
   * Extract features from one instance, store the extracted features in a
   * DecodeContext class.
   *
   *  @param[in]  inst    The instance.
   *  @param[io]  model   The model.
   *  @param[out] ctx     The decode context result.
   *  @param[in]  create  If create is true, create feature for new feature
   *                      in the model otherwise not create.
   */
  void extract_features(const Instance * inst,
      Model* mdl,
      DecodeContext* ctx,
      bool create = false);


  /**
   * Extract features from one instance, store the extracted features in the
   * nested DecodeContext.
   *
   *  @param[in]  inst    The instance
   *  @param[in]  create  If create is true, create feature for new
   *                          feature, otherwise not create.
   */
  virtual void extract_features(const Instance* inst, bool create = false);


  /**
   * Cache all the score for the certain instance. The cached results are
   * stored in a ScoreMatrix.
   *
   *  @param[in]  inst      the instance
   *  @param[in]  ctx       the decode context.
   *  @param[in]  use_avg   use to specify use average parameter
   *  @param[out] scm       the score matrix.
   */
  void calculate_scores(const Instance* inst,
      const Model* mdl,
      const DecodeContext* ctx,
      bool use_avg,
      ScoreMatrix* scm);


  /**
   * Cache all the score for the certain instance. The cached results are
   * stored in the nested ScoreMatrix.
   *
   *  @param[in]  inst      the instance
   *  @param[in]  use_avg   use to specify use average parameter
   */
  virtual void calculate_scores(const Instance* inst, bool use_avg);


  /**
   * Collect feature when given the tags index
   *
   * 
   *  @param[in]  model         The model.
   *  @param[in]  tagsidx       The tags index
   *  @param[out] vec           The output sparse vector
   */
  void collect_features(const math::Mat< math::FeatureVector* >& uni_features,
      const Model* mdl, const std::vector<int> & tagsidx,
      ltp::math::SparseVec & vec);


  /**
   * Collect correct and predicted features, mainly for the error-driven 
   * updating algorithm
   *
   *  @param[in/out]  inst      the instance
   */
  virtual void collect_correct_and_predicted_features(Instance * inst);


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

protected:
  //! The training flag.
  bool  __TRAIN__;

  //! The testing flag.
  bool  __TEST__;

  //! The dump flag.
  bool  __DUMP__;

  //! The timestamp.
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

  //! The collection of the training data.
  std::vector< Instance * > train_dat;

  //! The decode context
  DecodeContext* decode_context;

  //!
  ScoreMatrix* score_matrix;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_H__
