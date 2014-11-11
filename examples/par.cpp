#include <iostream>
#include <vector>

#include "ltp/parser_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    return -1;
  }

  void * engine = parser_create_parser(argv[1]);
  if (!engine) {
    return -1;
  }

  std::vector<std::string> words;
  std::vector<std::string> postags;

  words.push_back("一把手");  postags.push_back("n");
  words.push_back("亲自");    postags.push_back("d");
  words.push_back("过问");    postags.push_back("v");
  words.push_back("。");      postags.push_back("wp");

  std::vector<int>      heads;
  std::vector<std::string>  deprels;

  parser_parse(engine, words, postags, heads, deprels);

  for (int i = 0; i < heads.size(); ++ i) {
    std::cout << words[i] << "\t" << postags[i] << "\t"
              << heads[i] << "\t" << deprels[i] << std::endl;
  }

  parser_release_parser(engine);
  return 0;
}

