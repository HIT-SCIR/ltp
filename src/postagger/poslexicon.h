#ifndef __LTP_POSTAGGER_POSLEXICON_H__
#define __LTP_POSTAGGER_POSLEXICON_H__

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

namespace ltp {
namespace postagger {
class Poslexicon{

public:
  Poslexicon(){}
  ~Poslexicon(){}
  
  bool get(const std::string key,std::vector<int> & value) const{
    Lexiconmap_const_iter it = lexiconmap.find(key);
    if( it == lexiconmap.end() ){
      return false;
    }
    else{
      value = it->second;
      return true;
    }
  }
  
  void set(const std::string key,const std::vector<int> value){
    Lexiconmap_iter it = lexiconmap.find(key);
    if (it == lexiconmap.end() ){
      lexiconmap.insert(Lexiconmap::value_type(key, value));
    }
    else{
      std::vector <int> & origin = it->second;
      origin.insert( origin.begin(),value.begin(),value.end() );
      sort(origin.begin(),origin.end());
      origin.erase( unique(origin.begin(),origin.end()),origin.end() );
    }
  }

  void dump(){
     Lexiconmap_const_iter it = lexiconmap.begin();
     int lexicon_size;
     for(;it != lexiconmap.end();it++){
        std::cout<<it->first<<" ->";
        lexicon_size = (it->second).size();
        for(int i=0;i<lexicon_size;i++){
          std::cout<<" "<<(it->second)[i];
        }
        std::cout<<std::endl;
     }
  }
private:
  typedef std::map< std::string,std::vector<int> > Lexiconmap;
  typedef std::map<std::string, std::vector<int> >::iterator Lexiconmap_iter;
  typedef std::map<std::string, std::vector<int> >::const_iterator Lexiconmap_const_iter;
  Lexiconmap lexiconmap;
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_POSLEXICON_H__
