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
#include <ctime>
#include <string>
#include "time.hpp"
#include "segment_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 3) {
    std::cerr << "cws [baseline model path] [customized path] [lexicon_file](optional)" << std::endl;
    return 1;
  }

  void * engine = 0;
  if (argc == 3) {
    engine = segmentor_create_segmentor(argv[1], argv[2], NULL);
  } else if (argc == 4) {
    engine = segmentor_create_segmentor(argv[1], argv[2], argv[3]);
  }

  if (!engine) {
    return -1;
  }
  std::vector<std::string> words;
  std::string sentence;

  std::cerr << "TRACE: Model is loaded" << std::endl;
  double tm = ltp::utility::get_time();

  while (std::getline(std::cin, sentence, '\n')) {
    words.clear();
    if (sentence.size() == 0) { continue; }
    int len = segmentor_customized_segment(engine, sentence, words);
    for (int i = 0; i < len; ++ i) {
      std::cout << words[i];
      if (i+1 == len) std::cout <<std::endl;
      else std::cout<< "\t";
    }
  }
  tm = ltp::utility::get_time() - tm;
  std::cerr << "TRACE: cws-tm-consume "
            << tm
            << " seconds."
            << std::endl;

  segmentor_release_segmentor(engine);
  return 0;
}

