#ifndef __LTP_SEGMENTOR_SEGMENTOR_H__
#define __LTP_SEGMENTOR_SEGMENTOR_H__

#include "utils/cfgparser.hpp"
#include "segmentor/model.h"
#include "segmentor/decoder.h"
#include "segmentor/rulebase.h"

namespace ltp {
namespace segmentor {

class Segmentor {
public:
  Segmentor();
  Segmentor(ltp::utility::ConfigParser & cfg);
  ~Segmentor();

  void run();

private:
  /*
   * parse the configuration, return true on success, otherwise false
   *
   *  @param[in]  cfg         the config class
   *  @return     bool        return true on success, otherwise false
   */
  bool parse_cfg(ltp::utility::ConfigParser & cfg);

  /*
   * read instances from file and store them in train_dat
   *
   *  @param[in]  file_name   the filename
   *  @return     bool        true on success, otherwise false
   */
  bool read_instance( const char * file_name );

  /*
   * build tag sets, collect internal word map, record word frequency.
   */
  void build_configuration(void);

  /*
   *
   *
   */
  void build_feature_space(void);

  /*
   * the training process
   */
  void train(void);

  /*
   * the evaluating process
   */
  void evaluate(double &p, double &r, double &f);

  /*
   * the testing process
   */
  void test(void);

  /*
   * the dumping model process
   */
  void dump(void);

  /*
   * do feature selection by erasing the rare feature. create a new model
   * without rare feature (only witness a few times) according the original
   * model.
   *
   *  @param[in]  nr_updates  the number of update times
   *  @return     Model       the model without rare feature
   */
  Model * erase_rare_features(const int * nr_updates = NULL);
protected:
  /*
   * extract features from one instance,
   *
   */
  void extract_features(Instance * inst, bool create = false);

  /*
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

  /*
   * cache all the score for the certain instance.
   *
   *  @param[in/out]  inst      the instance
   *  @param[in]      use_avg   use to specify use average parameter
   */
  void calculate_scores(Instance * inst, bool use_avg);

  /*
   * collect feature when given the tags index
   *
   *  @param[in]    inst    the instance
   *  @param[in]    tagsidx the tags index
   *  @param[out]   vec     the output sparse vector
   */
  void collect_features(Instance * inst,
                        const std::vector<int> & tagsidx,
                        ltp::math::SparseVec & vec);


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
  bool  __TRAIN__;
  bool  __TEST__;
  bool  __DUMP__;

private:
  std::vector< Instance * > train_dat;

protected:
  Model *              model;
  Decoder *            decoder;
  rulebase::RuleBase * baseAll;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_H__
