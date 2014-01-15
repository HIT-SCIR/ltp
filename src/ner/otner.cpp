#include <iostream>
#include "cfgparser.hpp"
#include "logging.hpp"
#include "ner.h"

using namespace ltp::utility;
using namespace ltp::ner;

void usage(void) {
  std::cerr << "otcws - Training and testing suite for Named Entity Recognization"
            << std::endl;
  std::cerr << "Copyright (C) 2012-2014 HIT-SCIR" << std::endl;
  std::cerr << std::endl;
  std::cerr << "usage: ./otner <config_file>" << std::endl;
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

  NER engine(cfg);
  engine.run();
  return 0;
}
