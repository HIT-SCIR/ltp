/*
 * Single-threaded segmentor test program. The user input a line
 * of Chinese sentence an the program will output its segment
 * result.
 *
 *  @dependency package: tinythread - a portable c++ wrapper for
 *                       multi-thread library.
 *  @author:             LIU, Yijia
 *  @data:               2013-09-24
 *
 * This program is special designed for UNIX user, for get time
 * is not compilable under MSVC
 */
#include <iostream>
#include <sstream>
#include <ctime>
#include <cstring>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include "ltp/postag_dll.h"

double get_time(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

int main(int argc, char * argv[]) {
  if (argc < 2 || (0 == strcmp(argv[1], "-h"))) {
    std::cerr << "Example: ./pos_cmdline [model path]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "This program recieve input word sequence from stdin." << std::endl;
    std::cerr << "One sentence per line. Words are separated by space." << std::endl;
    std::cerr << std::endl;
    return 1;
  }

  void * engine = postagger_create_postagger(argv[1]);
  if (!engine) {
    std::cerr << "WARNINIG : Failed to load model." << std::endl;
    return -1;
  }

  std::string line;
  std::string word;
  std::vector<std::string> words;
  std::vector<std::string> postags;

  std::cerr << "TRACE: Model is loaded" << std::endl;
  double tm = get_time();

  while (std::getline(std::cin, line, '\n')) {
    std::stringstream S(line);
    words.clear();
    while (S >> word) { words.push_back(word); }

    if (words.size() == 0) { continue; }
    int len = postagger_postag(engine, words, postags);
    if (postags.size() != words.size()) {
      std::cerr << "WARNINIG: Number of postags is different from number of words"
                << std::endl;
    }

    for (int i = 0; i < len; ++ i) {
      std::cout << words[i] << "_" << postags[i];
      if (i+1 == len) std::cout <<std::endl;
      else std::cout<< "|";
    }
  }

  postagger_release_postagger(engine);

  tm = get_time() - tm;
  std::cerr << "TRACE: consume "
    << tm 
    << " seconds."
    << std::endl;

  return 0;
}

