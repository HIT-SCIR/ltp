#ifndef __LTP_UTIL_BITSET__
#define __LTP_UTIL_BITSET__

#include <iostream>
#include <string.h>
#include <vector>
#include "math.h"

namespace ltp {
namespace utility {

const int kBucketSize = int( sizeof(unsigned) ) * 8;
const int kN = int( log(kBucketSize)  / log(2) );

struct Bitset{
private:
  bool emptyflag;
  unsigned bits[4];
public:
  Bitset(){
    memset(bits,0,sizeof(bits));
    emptyflag = 1;
  }
  Bitset(int val){
    memset(bits,0,sizeof(bits));
    emptyflag = 1;
    set(val);
  }
  inline bool empty() const{
    return emptyflag;
  }
  inline bool allsetones(){
    memset(bits, 0xff, sizeof(bits));
    emptyflag = 1;
  }
  inline bool set(int val){
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    int bucket_index = val >> kN;
    int bucket_off = val & (kBucketSize - 1);
    if (bucket_index<0 || bucket_index >= bucket_cap){
      return false;
    }
    bits[bucket_index] |= (1<<bucket_off);
    emptyflag = 0;
    return true;
  }

  inline bool merge(const Bitset & other){
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    for(int i=0;i<bucket_cap;i++){
      bits[i] |= (other.bits[i]);
    }
    emptyflag &= (other.emptyflag);
    return true;
  }

  inline bool get(int val) const{
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    int bucket_index = val >> kN;
    int bucket_off = val & (kBucketSize - 1);
    if (bucket_index<0 || bucket_index >= bucket_cap){
      return false;
    }
    if(bits[bucket_index] & (1<<bucket_off)){
      return true;
    }
    return false;
  }
  inline std::vector<int> getbitones() const{
    std::vector<int> ones;
    int bucket_cap = sizeof(bits) / sizeof(unsigned);
    unsigned x,y;
    int curbit;
    for(int i=0;i<bucket_cap;i++){
      x = bits[i];
      while(x != 0){
        y = x&(x^(x-1));
        curbit = -1;
        while(y != 0){
          y >>= 1;
          curbit++;
        }//end while y!=0
        if(curbit != -1){
          curbit += (kBucketSize * i);
          ones.push_back(curbit);
        }//end if
        x = x&(x-1);
      }//end while x!=0
    }//end for
    return ones;
  }

};


}       //  end for namespace utility
}       //  end for namespace ltp
#endif    //  end for __LTP_UTIL_BITSET__
