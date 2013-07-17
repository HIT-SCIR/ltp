#include <iostream>
#include "cfgparser.hpp"
#include "logging.hpp"
#include "segmentor.h"

using namespace ltp::utility;
using namespace ltp::segmentor;

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        return -1;
    }

    ConfigParser cfg(argv[1]);

    if (!cfg) {
        ERROR_LOG("Failed to parse config file.");
        return -1;
    }

    Segmentor segmentor(cfg);
    segmentor.run();
    return 0;
}
