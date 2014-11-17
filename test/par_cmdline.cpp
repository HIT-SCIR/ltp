#include <iostream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cstring>
#include <string>
#include "time.hpp"
#include "parser_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    return -1;
  }

  void * engine = parser_create_parser(argv[1]);
  if (!engine) {
    return -1;
  }

  std::string line;
  std::string token;
  std::vector<std::string> words;
  std::vector<std::string> postags;
  std::vector<int> heads;
  std::vector<std::string> deprels;
  
  std::cerr << "TRACE: Model is loaded" << std::endl;
  double tm = ltp::utility::get_time();

  while (std::getline(std::cin, line, '\n')) {
    if (line.size() == 0) { continue; }

    std::stringstream S(line);
    words.clear();
    postags.clear();
    while (S >> token) {
      size_t npos = token.find_last_of("_");
      words.push_back(token.substr(0, npos));
      postags.push_back(token.substr(npos+ 1));
    }

    heads.clear();
    deprels.clear();
    parser_parse(engine, words, postags, heads, deprels);
    for (int i = 0; i < heads.size(); ++ i) {
      std::cout << words[i] << "\t" << postags[i] << "\t"
        << heads[i] << "\t" << deprels[i] << std::endl;
    }
    std::cout << std::endl;
  }

  tm = ltp::utility::get_time() - tm;
  std::cerr << "TRACE: par-tm-consume "
            << tm
            << " seconds."
            << std::endl;

  parser_release_parser(engine);
  return 0;
}

