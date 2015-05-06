#ifndef __HASHER_HPP__
#define __HASHER_HPP__


#include <cstring>

namespace ltp {
namespace utility {

struct __Default_CharArray_HashFunction {
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
    size_t hash = 5381;
    int c;
    const char* p = s.c_str();
    while (c = *p++)
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    return hash;
  }
};

}
}

#endif  //  end for __HASHER_HPP__
