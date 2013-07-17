/*
 * Graph Based Parser
 *
 * Author: Li, Zhenhua
 * Refactor:
 */

#include <iostream>
// #include "config.h"
#include "cfgparser.hpp"
#include "logging.hpp"
#include "parser.h"

using namespace std;
using namespace ltp::parser;

void usage(const char * msg) {
    // cerr << "lgdpj(lightweight)" << VERSION << " --- {lzh}@ir.hit.edu.cn" << endl;
    cerr << endl;
    cerr << msg << endl;

    cerr << "Usage: " << endl;
    cerr << endl;
    cerr << "  $./lgdpj config.txt" << endl;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage("");
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
