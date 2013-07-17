#include <iostream>
#include "cfgparser.hpp"
#include "logging.hpp"
#include "postagger.h"

using namespace ltp::utility;
using namespace ltp::postagger;

int main(int argc, const char * argv[]) {
    if (argc < 2) {
        return -1;
    }

    ConfigParser cfg(argv[1]);

    if (!cfg) {
        ERROR_LOG("Failed to parse config file.");
        return -1;
    }

    Postagger engine(cfg);
    engine.run();
    return 0;
}
