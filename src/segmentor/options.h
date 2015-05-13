#ifndef __LTP_SEGMENTOR_OPTIONS_H__
#define __LTP_SEGMENTOR_OPTIONS_H__

#include <iostream>
#include "framework/options.h"

namespace ltp {
namespace segmentor {

using framework::ModelOptions;
using framework::TestOptions;
using framework::DumpOptions;

struct TrainOptions: public framework::TrainOptions {
  bool dump_model_details;
  bool enable_incremental_training;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp

#endif    //  end for __LTP_SEGMENTOR_OPTIONS_H__
