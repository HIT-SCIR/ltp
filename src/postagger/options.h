#ifndef __LTP_POSTAGGER_OPTIONS_H__
#define __LTP_POSTAGGER_OPTIONS_H__

#include <iostream>
#include "framework/options.h"

namespace ltp {
namespace postagger {

using framework::ModelOptions;
using framework::TrainOptions;
using framework::TestOptions;
using framework::DumpOptions;

extern ModelOptions model_opt;
extern TrainOptions train_opt;
extern TestOptions  test_opt;
extern DumpOptions  dump_opt;

}       //  end for namespace postagger
}       //  end for namespace ltp

#endif    //  end for __LTP_POSTAGGER_OPTIONS_H__
