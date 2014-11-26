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
#include "ltp/segment_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cerr << "cws [model path] [lexicon_file]" << std::endl;
    return 1;
  }

  std::vector<std::string> words;
  std::string sentence;


  void * segmentor = segmentor_create_segmentor(argv[1],argv[2],NULL);
  if (!segmentor) {
    std::cout<<"failed to init"<<std::endl;
    return 0;
  }

  while(std::getline(std::cin, sentence, '\n')){
    words.clear();
    int len = segmentor_customized_segment(segmentor, sentence, words);
    for (int j = 0; j < len; ++ j) {
      std::cout << words[j];
      if (j+1 == len) std::cout <<std::endl;
      else std::cout<< "\t";
    }
  }
  segmentor_release_segmentor(segmentor);
  return 0;
}

