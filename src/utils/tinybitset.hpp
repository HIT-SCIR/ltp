#ifndef __TINYBITSET__
#define __TINYBITSET__

#include <iostream>
#include <map>
#include <vector>
#include <string.h>
#include <algorithm>

namespace ltp {
namespace utility {

struct Bitset{
  bool nonemptyflag;
  unsigned bits[4];
  Bitset(){
    memset(bits,0,sizeof(bits));
    nonemptyflag = 0;
  }
  Bitset(int val){
    memset(bits,0,sizeof(bits));
    nonemptyflag = 0;
    set(val);
  }
  inline bool isnotempty() const{
    return nonemptyflag;
  }
  inline bool set(int val){
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    int bucket_size = int( sizeof(unsigned) ) * 8;
    int bucket_index = val / bucket_size;
    int bucket_off = val % bucket_size;
    if (bucket_index<0 || bucket_index >= bucket_cap){
      return false;
    }
    bits[bucket_index] |= (1<<bucket_off);
    nonemptyflag = 1;
    return true;
  }
  inline bool merge(Bitset & other){
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    for(int i=0;i<bucket_cap;i++){
      bits[i] |= (other.bits[i]);
    }
    nonemptyflag |= (other.nonemptyflag);
    return true;
  }
  inline bool get(int val) const{
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    int bucket_size = int( sizeof(unsigned) ) * 8;
    int bucket_index = val / bucket_size;
    int bucket_off = val % bucket_size;
    if (bucket_index<0 || bucket_index >= bucket_cap){
      return false;
    }
    if(bits[bucket_index] & (1<<bucket_off)){
      return true;
    }
    return false;
  }
  inline void debug() const{
    int idxrange = int( sizeof(bits) ) * 8;
    for(int i=0;i<idxrange;i++){
      if(get(i)){
        std::cout<<i<<" ";
      }
    }
    std::cout<<std::endl;
  }
  /*inline std::vector<int> debug() const{
      std::vector<int> tmp;
      for(int i=0;i<128;i++){
        if(get(i)){
          tmp.push_back(i);
        }
      }
      return tmp;
  }*/

};


}       //  end for namespace utility
}       //  end for namespace ltp
#endif    //  end for __TINYBITSET__
