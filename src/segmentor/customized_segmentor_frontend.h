#ifndef __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_FRONTEND_H__
#define __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_FRONTEND_H__

#include "segmentor/segmentor_frontend.h"
//#include "segmentor/customized_options.h"

namespace ltp {
namespace segmentor {

/**
 * The class CustomizedSegmentor inherit from the class Segmentor.
 * It is designed for customized segment, which means users can 
 * use two models, baseline model and customized model to segment
 * words. Most of the function responsibility between Segmentor and
 * Customized is the same, except for that CustomziedSegmentor takes
 * two parts, the baseline and the customized, as input.
 */
class CustomizedSegmentorFrontend : public SegmentorFrontend {
protected:
  Model* bs_model;
  framework::ViterbiFeatureContext bs_ctx;  //! The decode context
  std::string bs_model_file;
  bool good;

public:
  //! Learning model constructor.
  CustomizedSegmentorFrontend(const std::string& reference_file,
    const std::string& holdout_file,
    const std::string& model_name,
    const std::string& baseline_model_file,
    const std::string& algorithm,
    const size_t& max_iter,
    const size_t& rare_feature_threshold);

  CustomizedSegmentorFrontend(const std::string& input_file,
    const std::string& model_file,
    const std::string& baseline_model_file,
    bool evaluate);

  //!
  ~CustomizedSegmentorFrontend();

protected:
  bool load_baseline_model();
  void build_configuration(void);
  void extract_features(const Instance& inst, bool create);
  void calculate_scores(const Instance& inst, bool avg);
  void collect_features(const Instance& inst);
  void update(const Instance& inst, math::SparseVec& updated_features);
  void setup_lexicons();
  void clear_context();
};

} // end for namespace segmentor
} // end for namespace ltp

#endif // end for __LTP_CUSTOMIZED_SEGMENTOR_FRONTEND_H__
