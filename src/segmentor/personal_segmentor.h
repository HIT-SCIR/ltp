#ifndef __LTP_SEGMENTOR_PERSONAL_SEGMENTOR_H__
#define __LTP_SEGMENTOR_PERSONAL_SEGMENTOR_H__

#include "segmentor/segmentor.h"

namespace ltp {
namespace segmentor {

class Personal_Segmentor : public Segmentor{
public:
  Personal_Segmentor();
  Personal_Segmentor(ltp::utility::ConfigParser & cfg);
  ~Personal_Segmentor();

protected:
  bool parse_cfg(ltp::utility::ConfigParser & cfg);
  void build_configuration(void);
  void extract_features(Instance * inst, bool create = false);
  void build_feature_space(void);
  void calculate_scores(Instance * inst, bool use_avg);
  void collect_features(Instance * inst, const std::vector<int> & tagsidx, math::SparseVec & vec, math::SparseVec & personal_vec);
  Model * erase_rare_features(const int * feature_updated_times);
  void train(void);
  void evaluate(double &p, double &r, double &r);
  void test(void);
  void dump();
protected:
  Model * personal_model;
};

} // end for namespace segmentor
} // end for namespace ltp

#endif // end for __LTP_SEGMENTOR_PERSONAL_SEGMENTOR_H__
