/**
 * A library for mapping string to user specified type.
 * Modify from @file CharArrayHashFunc.h and @file CharArrayEqualFunc.h
 *
 *  @author:    Mihai Surdeanu
 *  @modifier:  LI, Zhenhua
 *  @modifier:  LIU, Yijia
 */

#ifndef __STRING_MAP_HPP__
#define __STRING_MAP_HPP__

#include <string.h>
#include <stdlib.h>
#include "unordered_map.hpp"
#include "hasher.hpp"

namespace ltp {
namespace utility {

template <class T>
class StringMap {
public:
  typedef std::unordered_map<const char *, T,
          __Default_CharArray_HashFunction,
          __Default_CharArray_EqualFunction> internal_map_t;

  typedef typename internal_map_t::iterator       iterator;
  typedef typename internal_map_t::const_iterator const_iterator;

  StringMap() {
  }

  ~StringMap() {
    clear();
  }

  void clear() {
    const char * last = NULL;
    for (iterator it = _map.begin(); it != _map.end(); ++ it) {
      if (last != NULL) {
        free( (void *)last );
      }
      last = it->first;
    }

    if (last != NULL) {
      free( (void *)last );
    }
    _map.clear();
  }

  bool set( const char * key, const T &val ) {
    if (contains(key)) {
      return false;
    }

    int len = 0;
    for (; key[len] != 0; ++ len);

    char * new_key = (char *) malloc( (len + 1) * sizeof(char) );
    for (int i = 0; i < len; ++ i) {
      new_key[i] = key[i];
    }

    new_key[len] = 0;
    _map[new_key] = val;
    return true;
  }

  void unsafe_set(const char * key, const T &val ) {
    int len = 0;
    for (; key[len] != 0; ++ len);

    char * new_key = (char *) malloc( (len + 1) * sizeof(char) );
    for (int i = 0; i < len; ++ i) {
      new_key[i] = key[i];
    }

    new_key[len] = 0;
    _map[new_key] = val;
  }

  bool overwrite( const char * key, const T &val ) {
    if (contains(key)) {
      iterator it = _map.find(key);
      it->second = val;
      return true;
    } else {
      return set(key, val);
    }
    return false;
  }

  bool get( const char * key, T& val) const {
    const_iterator it;
    if ((it = _map.find(key)) != _map.end()) {
      val = it->second;
      return true;
    }
    return false;
  }

  T* get(const char * key) {
    iterator it = _map.find(key);
    if (it != _map.end()) {
      return &(it->second);
    }
    return NULL;
  }

  void unsafe_get( const char * key, T& val) {
    val = _map.find(key)->second;
  }

  bool contains( const char * key ) const {
    if (_map.find(key) != _map.end()) {
      return true;
    }
    return false;
  }

  size_t size() const {
    return _map.size();
  }

  bool empty() const {
    return _map.empty();
  }

  const_iterator begin() const {
    return _map.begin();
  }

  const_iterator end() const {
    return _map.end();
  }

  iterator mbegin() {
    return _map.begin();
  }

  iterator mend() {
    return _map.end();
  }

protected:
  internal_map_t _map;
};  // end for class StringMap

}   // end for namespace utility
}   // end for namespace ltp


#endif  // end for __STRING_MAP_HPP__
