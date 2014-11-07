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
  void build_feature_space(void);
  void calculate_scores(Instance * inst, bool use_avg);
  void collect_features(Instance * inst, const std::vector<int> & tagsidx, math::SparseVec & vec, math::SparseVec & personal_vec);
  Model * erase_rare_features(const int * feature_updated_times);
  void train(void);
  void evaluate(double &p, double &r, double &f);
  void test(void);
  void dump();

protected:
  Model * baseline_model;

  CustomizedTrainOptions train_opts;
  CustomizedTestOptions test_opts;
  //CustomizedDumpOptions dump_opts;
};

} // end for namespace segmentor
} // end for namespace ltp

#endif // end for __LTP_SEGMENTOR_PERSONAL_SEGMENTOR_H__
