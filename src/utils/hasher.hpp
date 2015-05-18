#ifndef __HASHER_HPP__
#define __HASHER_HPP__


#include <cstring>

namespace ltp {
namespace utility {

static size_t __hash(const char* s) {
  size_t hash = 0;
  while (*s) {
    hash = hash * 101 + *s ++;
  }
  return size_t(hash);
}

struct __Default_CharArray_HashFunction {
  size_t operator () (const char* s) const {
    return __hash(s);
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
    return __hash(s.c_str());
  }
};

}
}

#endif  //  end for __HASHER_HPP__
