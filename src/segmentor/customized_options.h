#ifndef __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_OPTIONS_H__
#define __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_OPTIONS_H__

#include "options.h"

namespace ltp {
namespace segmentor {

struct CustomizedTrainOptions : public TrainOptions {
  //! The name of the baseline model. aka. model from phase#1
  std::string baseline_model_name;
};

struct CustomizedTestOptions : public TestOptions {
  //! The name of the baseline model. aka. model from phase#1
  std::string baseline_model_file;
};

struct CustomizedDumpOptions : public DumpOptions {
};

} //  end for namespace segmentor
} //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_CUSTOMIZED_SEGMENTOR_OPTIONS_H__
