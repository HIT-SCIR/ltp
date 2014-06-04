#ifndef __LTP_UTILS_BITSET__
#define __LTP_UTILS_BITSET__

#include <iostream>
#include <string.h>
#include <vector>

namespace ltp {
namespace utility {

const int kBucketSize = int( sizeof(unsigned) ) * 8;
const int kBucketCap = 4;

// A parallize method for calculate number of bits in a number. Adopt this from
// http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
const int kN1 = (kBucketSize - 1) - (((kBucketSize - 1) >> 1) & 0x55555555);
const int kN2 = (kN1 & 0x33333333) + ((kN1 >> 2) & 0x33333333);
const int kN = (((kN2 + (kN2 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;;

struct Bitset {
private:
  bool emptyflag;
  unsigned bits[kBucketCap];
public:
  // Constructor
  Bitset() {
    memset(bits,0,sizeof(bits));
    emptyflag = 1;
  }

  Bitset(int val) {
    memset(bits,0,sizeof(bits));
    emptyflag = 1;
    set(val);
  }

  inline bool empty() const {
    return emptyflag;
  }

  inline void allsetones() {
    memset(bits, 0xff, sizeof(bits));
    emptyflag = 1;
  }

  inline bool set(int val) {
    int bucket_index = val >> kN;
    int bucket_off = val & (kBucketSize - 1);
    if (bucket_index<0 || bucket_index >= kBucketCap){
      return false;
    }
    bits[bucket_index] |= (1<<bucket_off);
    emptyflag = 0;
    return true;
  }

  /**
   * Merge two Bitset together.
   *
   * e.g.:
   *
   *  Bitset mask1(0x02), mask2(0x03);
   *  mask1.merge(mask2);
   *  std::cout << mask1.bitset[0] << std::endl; // => 3
   *
   *  @param[in]  other   The other bitset
   *  @return     bool    ?
   */
  inline bool merge(const Bitset & other) {
    for(int i = 0; i < kBucketCap; ++ i) {
      bits[i] |= (other.bits[i]);
    }
    emptyflag &= (other.emptyflag);
    return true;
  }

  /**
   * Get the ith bit of the bitset
   *
   * e.g.:
   *
   *  Bitset mask1(0x02);
   *  std::cout << mask1.get(0) << std::endl; // => false
   *  std::cout << mask1.get(1) << std::endl; // => true
   *
   *  @param[in]  val   Specify ith bit
   *  @return     bool  Return true on ith bit is true, otherwise false.
   */
  inline bool get(int val) const {
    int bucket_index = val >> kN;
    int bucket_off = val & (kBucketSize - 1);
    if (bucket_index<0 || bucket_index >= kBucketCap){
      return false;
    }
    if(bits[bucket_index] & (1<<bucket_off)) {
      return true;
    }
    return false;
  }

  inline std::vector<int> getbitones() const{
    std::vector<int> ones;
    unsigned x,y;
    int curbit;
    for(int i = 0; i < kBucketCap; ++ i){
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
#endif  //  end for __LTP_UTILS_BITSET__
