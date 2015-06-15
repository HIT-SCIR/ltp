// This function implements a vector specially designed for storing
// string. The mainly technique used in this function is the memory
// pool.
#ifndef __STRING_VECTOR_HPP__
#define __STRING_VECTOR_HPP__

#include <iostream>
#include <stdio.h>
#include <string.h>

namespace ltp {
namespace utility {

class StringVec {
public:
  // allocator for string vector, clear the buffer
  // initalize the length.
  StringVec() :
    _pool(0),
    _len_pool(0),
    _cap_pool(0),
    _index(0),
    _len_index(0),
    _cap_index(0) {}

  ~StringVec() {
    if (_pool) {
      delete [](_pool);
    }

    if (_index) {
      delete [](_index);
    }
  }

  int push_back(const std::string & str) {
    return push_back(str.c_str());
  }


  // push the string str to StringVector
  int push_back(const char * str) {
    size_t len = strlen(str) + 1;
    size_t new_len;

    if (_cap_pool <= (new_len = (_len_pool + len))) {
      _cap_pool = (new_len << 1);
      char * new_pool = new char[_cap_pool];
      if (_pool) {
        memcpy(new_pool, _pool, sizeof(char) * _len_pool);
        delete [](_pool);
      }

      _pool = new_pool;
    }

    if (_cap_index <= (new_len = (_len_index + 1))) {
      _cap_index = (new_len << 1);
      int * new_index = new int[_cap_index];
      if (_index) {
        memcpy(new_index, _index, sizeof(int) * _len_index);
        delete [](_index);
      }

      _index = new_index;
    }

    memcpy(_pool + _len_pool, str, len);
    _index[_len_index] = _len_pool;

    _len_pool += len;
    ++ _len_index;

    return _len_index;
  }

  const char * operator [](const size_t& i) const {
    if (i >= _len_index) {
      return 0;
    }
    return _pool + _index[i];
  }

  size_t size() const { return _len_index; }

  void clear() {
    _len_pool = 0;
    _len_index = 0;
  }

  void debug() {
    printf("%p ", _pool);
    for (size_t i = 0; i < _len_pool; ++ i) {
      putchar(_pool[i]);
    }
    putchar('\n');
    for (size_t i = 0; i < _len_index; ++ i) {
      printf("%p %s\n", _pool + _index[i], _pool + _index[i]);
    }
  }

private:
  char * _pool;
  size_t _len_pool;
  size_t _cap_pool;

  int * _index;
  size_t _len_index;
  size_t _cap_index;
};

}
}

#endif  //  end for __STRING_VECTOR_HPP__
