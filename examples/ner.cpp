#include <iostream>
#include <vector>

#include "ltp/ner_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cerr << "usage: ./ner [model_path]" << std::endl;
    return -1;
  }

  void * engine = ner_create_recognizer(argv[1]);
  if (!engine) {
    std::cerr << "failed to load model" << std::endl;
    return -1;
  }

  std::vector<std::string> words;
  std::vector<std::string> postags;

  words.push_back("中国");    postags.push_back("ns");
  words.push_back("国际");    postags.push_back("n");
  words.push_back("广播");    postags.push_back("n");
  words.push_back("电台");    postags.push_back("n");
  words.push_back("创办");    postags.push_back("v");
  words.push_back("于");      postags.push_back("p");
  words.push_back("1941年");  postags.push_back("m");
  words.push_back("12月");    postags.push_back("m");
  words.push_back("3日");     postags.push_back("m");
  words.push_back("。");      postags.push_back("wp");

  std::vector<std::string>  tags;

  ner_recognize(engine, words, postags, tags);

  for (int i = 0; i < tags.size(); ++ i) {
    std::cout << words[i] << "\t" << postags[i] << "\t" << tags[i] << std::endl;
  }

  ner_release_recognizer(engine);
  return 0;
}

