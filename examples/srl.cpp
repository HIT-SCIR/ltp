#include <iostream>
#include <vector>

#include "ltp/srl_dll.h"

int main(int argc, char * argv[]) {
  if (argc < 2) {
    return -1;
  }

  srl_load_resource(argv[1]);

  std::vector<std::string> words;
  std::vector<std::string> postags;
  std::vector<std::string> nes;
  std::vector<std::pair<int,std::string> > parse;
  std::vector< std::pair< int, std::vector< std::pair<std::string, std::pair< int, int > > > > > srl;
  words.push_back("一把手");  postags.push_back("n"); nes.push_back("O"); parse.push_back(make_pair(2,"SBV"));
  words.push_back("亲自");    postags.push_back("d"); nes.push_back("O"); parse.push_back(make_pair(2,"ADV")); 
  words.push_back("过问");    postags.push_back("v"); nes.push_back("O"); parse.push_back(make_pair(-1,"HED"));
  words.push_back("。");      postags.push_back("wp");nes.push_back("O"); parse.push_back(make_pair(2,"WP"));

  srl_dosrl(words,postags,parse,srl);

  for(int i = 0;i<srl.size();++i) {
    std::cout<<srl[i].first<<":"<<std::endl;
    for(int j = 0;j<srl[i].second.size();++j) {
      std::cout<<"\ttype = "<<srl[i].second[j].first
               <<" beg = "<<srl[i].second[j].second.first 
               <<" end = "<<srl[i].second[j].second.second
               <<std::endl; 
    }
  }

  srl_release_resource();
  return 0;
}

