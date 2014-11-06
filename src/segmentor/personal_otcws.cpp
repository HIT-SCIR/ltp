#include <iostream>
#include "utils/cfgparser.hpp"
#include "utils/logging.hpp"
#include "segmentor/personal_segmentor.h"

using namespace ltp::utility;
using namespace ltp::segmentor;

void usage(void) {
  std::cerr << "otcws - Training and testing suite for Chinese Word segmentation"
            << std::endl;
  std::cerr << "Copyright (C) 2012-2014 HIT-SCIR" << std::endl;
  std::cerr << std::endl;
  std::cerr << "usage: ./otcws <config_file>" << std::endl;
  std::cerr << std::endl;
}

int main(int argc, const char * argv[]) {
  if (argc < 2 || (argv[1][0] == '-' && argv[1][1] == 'h')) {
    usage();
    return -1;
  }

  ConfigParser cfg(argv[1]);

  if (!cfg) {
    ERROR_LOG("Failed to parse config file.");
    return -1;
  }

  Personal_Segmentor segmentor(cfg);
  segmentor.run();
  return 0;
}
