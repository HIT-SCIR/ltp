#ifndef __LTP_SEGMENTOR_OPTIONS_H__
#define __LTP_SEGMENTOR_OPTIONS_H__

#include <iostream>

namespace ltp {
namespace segmentor {

struct ModelOptions {
  std::string   model_file;
};

struct TrainOptions {
  std::string       train_file;
  std::string       holdout_file;
  std::string       model_file;
  std::string       algorithm;
  int               max_iter;
  int               display_interval;
  int               rare_feature_threshold;
  bool              enable_incremental_training;
};

struct TestOptions {
  std::string       test_file;
  std::string       model_file;
  std::string       lexicon_file;
};

struct DumpOptions {
  std::string       model_file;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp

#endif    //  end for __LTP_SEGMENTOR_OPTIONS_H__
