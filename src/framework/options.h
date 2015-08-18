#ifndef __LTP_FRAMEWOKR_OPTIONS_H__
#define __LTP_FRAMEWOKR_OPTIONS_H__

#include <iostream>

namespace ltp {
namespace framework {

struct ModelOptions {
  std::string   model_file;
};

struct TrainOptions {
  std::string train_file;
  std::string holdout_file;
  std::string model_name;
  std::string algorithm;
  size_t max_iter;
  size_t rare_feature_threshold;
};

struct TestOptions {
  std::string test_file;
  std::string model_file;
  std::string lexicon_file;
  bool evaluate;
  bool sequence_prob;
  bool marginal_prob;
};

struct DumpOptions {
  std::string   model_file;
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWOKR_OPTIONS_H__
