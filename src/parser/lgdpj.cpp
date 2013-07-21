/*
 * Graph Based Parser
 *
 * Author: LI, Zhenghua
 * Author: LIU, Yijia
 */
#include <iostream>
#include "cfgparser.hpp"
#include "logging.hpp"
#include "parser.h"

using namespace ltp::parser;

void usage(void) {
    std::cerr << "lgdpj - Training and testing suite for Chinese Dependency Parsering" << std::endl;
    std::cerr << "Copyright (C) 2012-2013 HIT-SCIR" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Usage: $./lgdpj <config_file>" << std::endl;
    std::cerr << std::endl;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return -1;
    }

    ConfigParser cfg(argv[1]);

    if (!cfg) {
        ERROR_LOG("Failed to parse config file.")
        return -1;
    }

    Parser parser(cfg);
    parser.run();
    return 0;
}
