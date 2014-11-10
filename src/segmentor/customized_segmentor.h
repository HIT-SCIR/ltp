#ifndef __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_H__
#define __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_H__

#include "segmentor/segmentor.h"
#include "segmentor/customized_options.h"

namespace ltp {
namespace segmentor {

namespace utils = ltp::utility;

class CustomizedSegmentor : public Segmentor{
public:
  //!
  CustomizedSegmentor();

  //!
  CustomizedSegmentor(utils::ConfigParser & cfg);

  //!
  ~CustomizedSegmentor();

protected:
  //!
  bool parse_cfg(utils::ConfigParser & cfg);

  //!
  void build_configuration(void);

  //!
  void extract_features(Instance * inst, bool create = false);

  //!
  void calculate_scores(Instance * inst, bool use_avg);

  //!
  void collect_correct_and_predicted_features(Instance * inst);

  //
  bool train_setup(void);

  //!
  void train_passive_aggressive(int nr_errors);

  //!
  void train_averaged_perceptron(void);

  //!
  int get_timestamp(void);

  //!
  void evaluate(double &p, double &r, double &f);

  bool test_setup(void);

protected:
  Model * baseline_model;

  //!
  math::SparseVec baseline_correct_features;

  //!
  math::SparseVec baseline_predicted_features;

  //!
  math::SparseVec baseline_updated_features;

  //!
  math::Mat< math::FeatureVector * > baseline_uni_features;

  CustomizedTrainOptions train_opts;
  CustomizedTestOptions test_opts;
  //CustomizedDumpOptions dump_opts;
};

} // end for namespace segmentor
} // end for namespace ltp

#endif // end for __LTP_SEGMENTOR_PERSONAL_SEGMENTOR_H__
