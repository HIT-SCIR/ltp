// This code is modified from @file utf.h in zpar project
//
//  @author: ZHANG, Yue <http://www.sutd.edu.sg/yuezhang.aspx>
//  @modifier: LIU, Yijia <yjliu@ir.hit.edu.cn>
#ifndef __CODECS_HPP__
#define __CODECS_HPP__

#include <iostream>
#include <string>
#include <vector>
#include "unicode.tab"

namespace ltp {
namespace strutils {
namespace codecs {
enum { UTF8, GBK };

class iterator {
public:
  iterator(const iterator & other)
    : str(other.str), rep(other.rep),
    encoding(other.encoding), healthy(other.healthy) {
  }

  iterator(const char * _str, int _encoding=UTF8)
    : str(_str), rep(0, 0), encoding(_encoding), healthy(true) {
    find_second_by_first(rep);
  }

  iterator(const std::string & _str, int _encoding=UTF8)
    : str(_str.c_str()), rep(0, 0), encoding(_encoding), healthy(true) {
    find_second_by_first(rep);
  }

  iterator& operator ++ () {
    rep.first = rep.second;
    if (str[rep.first]) {
      find_second_by_first(rep);
    }
    return (*this);
  }

  iterator& operator ++ (int dummy) {
    rep.first = rep.second;
    if (str[rep.first]) {
      find_second_by_first(rep);
    }
    return (*this);
  }

  std::pair<int, int> operator * () const {
    return rep;
  }

  const std::pair<int, int> * operator->() const {
    return (&rep);
  }

  bool has_next() {
    return (str[rep.second] != 0);
  }

  std::pair<int, int> next() {
    std::pair<int, int> retval(rep.second, 0);
    find_second_by_first(retval);
    return retval;
  }

  bool is_end() {
    return (str[rep.first] == 0);
  }

  bool is_good() {
    return healthy;
  }

  int encoding;
  bool healthy;
  const char * str;
  std::pair<int, int> rep;

private:
  void find_second_by_first(std::pair<int, int>& payload) {
    if (str[payload.first] == 0) {
      return;
    }
    if (encoding == UTF8) {
      if ((str[payload.first]&0x80) == 0) {
        payload.second = payload.first + 1;
      } else if ((str[payload.first]&0xE0) == 0xC0) {
        payload.second = payload.first + 2;
      } else if ((str[payload.first]&0xF0) == 0xE0) {
        payload.second = payload.first + 3;
      } else if ((str[payload.first]&0xF8) == 0xF0) {
        payload.second = payload.first + 4;
      } else {
        healthy = false;
      }
    } else if (encoding == GBK) {
      if ((str[payload.first]&0x80) == 0) {
        payload.second = payload.first + 1;
      } else {
        payload.second = payload.first + 2;
      }
    }
  }
};

inline int decode(const std::string & s, std::vector<std::string>& chars,
  int encoding=UTF8) {
  int len=0;
  chars.clear();

  iterator itx = iterator(s, encoding);
  for (; itx.is_good() && !itx.is_end();++ itx, ++ len) {
    chars.push_back(s.substr(itx->first, itx->second- itx->first));
  }
  if (false == itx.is_good()) {
    std::cerr << "Warning: in codecs.hpp decode: string '" << s
      << "' is not encoded in unicode utf-8" << std::endl;
    chars.clear();
    return -1;
  }
  return len;
}


inline size_t length(const std::string & s, int encoding=UTF8) {
  size_t len = 0;
  for (iterator itx = iterator(s, encoding); itx.is_good() && !itx.is_end();++ itx, ++ len);
  return len;
}


inline bool initial(const std::string & s, 
    std::string & ch, int encoding=UTF8) {
  if (s=="") {
    return false;
  }

  iterator itx = iterator(s, encoding);
  if (false == itx.is_good()) {
    return false;
  } else {
    ch = s.substr(itx->first, itx->second);
  }
  return true;
}


inline bool tail(const std::string & s,
    std::string & ch, int encoding=UTF8) {
  int first = 0, second = 0;
  if (0 == s.size()) {
    return false;
  }

  iterator itx(s, encoding);
  for (; itx.is_good() && !itx.is_end(); ++ itx) {
    first = itx->first;
    second = itx->second;
  }

  if (!itx.is_good()) {
    return false;
  } else {
    ch = s.substr(first, second - first);
    return true;
  }
  return false;
}


inline bool isclear(const std::string & s, int encoding=UTF8) {
  unsigned idx = 0;
  if (encoding != UTF8 && encoding != GBK) {
    return false;
  } else {
    iterator itx(s, encoding);
    for (; itx.is_good() && !itx.is_end(); ++ itx);
    if (!itx.is_good()) { return false; }
  }
  return true;
}

inline bool is_unicode_punctuation(const std::string& str) {
  if (length(str) != 1) {
    return false;
  }
  size_t code = 0;
  if(!(str[0]&0x80))      // 0xxxxxxx
    code = str[0];
  else if ((str[0]&0xE0)==0xC0)  // 110xxxxx
    code = (((str[0]&0x1F)<<6) | (str[1]&0x3F));
  else if ((str[0]&0xF0)==0xE0)  // 1110xxxx
    code = (((str[0]&0x0F)<<12) | ((str[1]&0x3F)<<6) | (str[2]&0x3F));
  else if ((str[0]&0xF8)==0xF0)
    code = (((str[0]&0x07)<<18) | ((str[1]&0x3F)<<12) | ((str[2]&0x3F)<<6) | (str[3]&0x3F));
  else
    return false;

  // UTF8 General Punctuation.
  for (size_t i = 0; i < UNICODE_PUNCTUATION_SIZE; ) {
    if (code >= UNICODE_PUNCTUATION[i] && code <= UNICODE_PUNCTUATION[i+1]) {
      return true;
    }
    i += 2;
  }
  return false;
}

}       //  end for namespace codecs
}       //  end for namespace strutils
}       //  end for namespace ltp
#endif  //  end for __CODECS_HPP__

