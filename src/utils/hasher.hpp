#ifndef __HASHER_HPP__
#define __HASHER_HPP__

#if defined(_MSC_VER)
  #include <hash_map>
#endif

#include <cstring>

namespace ltp {
namespace utility {

struct __Default_CharArray_HashFunction
#if defined(_MSC_VER)
  : public stdext::hash_compare<const char *>
#endif
{
  size_t operator () (const char* s) const {
    unsigned int hash = 0;
    while (*s) {
      hash = hash * 101 + *s ++;
    }
    return size_t(hash);
  }

  bool operator() (const char* s1, const char* s2) const {
    return (strcmp(s1, s2) < 0);
  }
};

struct __Default_CharArray_EqualFunction {
  bool operator () (const char* s1, const char* s2) const {
    return (strcmp(s1, s2) == 0);
  }
};

struct __Default_String_HashFunction {
  size_t operator()(const std::string& s) const {
    unsigned int _seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int _hash = 0;
    for(std::size_t i = 0; i < s.size(); i++) {
      _hash = (_hash * _seed) + s[i];
    }
    return size_t(_hash);
  }
};


}
}

#endif  //  end for __HASHER_HPP__
