/*
 * This code is modified from @file utf.h in @project zpar<
 * 
 *  @author: ZHANG, Yue <http://www.sutd.edu.sg/yuezhang.aspx>
 *  @modifier: LIU, Yijia <yjliu@ir.hit.edu.cn>
 */
#ifndef __CODECS_HPP__
#define __CODECS_HPP__

#include <iostream>
#include <vector>

namespace ltp {
namespace strutils {
namespace codecs {
enum { UTF8, GBK };

class iterator {
public:
  iterator(const iterator & other)
    : str(other.str), payload(other.payload),
    encoding(other.encoding), healthy(other.healthy) {
  }

  iterator(const char * _str, int _encoding=UTF8) 
    : str(_str), payload(0, 0), encoding(_encoding), healthy(true) {
    find_second_by_first();
  }

  iterator(const std::string & _str, int _encoding=UTF8)
    : str(_str.c_str()), payload(0, 0), encoding(_encoding), healthy(true) {
    find_second_by_first();
  }

  iterator& operator ++ () {
    payload.first = payload.second;
    if (str[payload.first]) {
      find_second_by_first();
    }
  }

  iterator& operator ++ (int dummy) {
    payload.first = payload.second;
    if (str[payload.first]) {
      find_second_by_first();
    }
  }

  std::pair<int, int> operator * () const {
    return payload;
  }

  const std::pair<int, int> * operator->() const {
    return (&payload);
  }

  bool is_end() {
    return (str[payload.first] == 0);
  }

  bool is_good() {
    return healthy;
  }

  int encoding;
  bool healthy;
  const char * str;
  std::pair<int, int> payload;

private:
  void find_second_by_first() {
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


inline
int decode(const std::string & s, 
    std::vector<std::string>& chars, int encoding=UTF8) {
  unsigned long int idx=0;
  unsigned long int len=0;
  chars.clear();

  if (encoding == UTF8) {
    while (idx<s.length()) {
      if ((s[idx]&0x80)==0) {
        chars.push_back(s.substr(idx, 1));
        ++len;
        ++idx;
      } else if ((s[idx]&0xE0)==0xC0) {
        chars.push_back(s.substr(idx, 2));
        ++len;
        idx+=2;
      } else if ((s[idx]&0xF0)==0xE0) {
        chars.push_back(s.substr(idx, 3));
        ++len;
        idx+=3;
      } else if ((s[idx]&0xF8)==0xF0) {
        chars.push_back(s.substr(idx, 4));
        ++len;
        idx+=4;
      } else {
        std::cerr << "Warning: in codecs.hpp decode: string '" << s 
          << "' is not encoded in unicode utf-8" << std::endl;
        ++len;
        ++idx;
      }
    }
  } else if (encoding == GBK) {
    while (idx<s.length()) {
      if ((s[idx]&0x80)==0) {
        chars.push_back(s.substr(idx, 1));
        ++ len;
        ++ idx;
      } else {
        chars.push_back(s.substr(idx, 2));
        ++ len;
        idx += 2;
      }
    }
  } else {
    return 0;
  }

  return len;
}


inline
int length(const std::string & s, int encoding=UTF8) {
  unsigned int len = 0;
  for (iterator itx = iterator(s, encoding); itx.is_good() && !itx.is_end();++ itx, ++ len);
  return len;
}


inline
bool initial(const std::string & s, 
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


inline
bool tail(const std::string & s,
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


inline
bool isclear(const std::string & s, int encoding=UTF8) {
  int idx = 0;
  if (encoding == UTF8) {
    while (idx<s.length()) {
      if ((s[idx]&0x80)==0) {
        ++idx;
      } else if ((s[idx]&0xE0)==0xC0) {
        idx+=2;
      } else if ((s[idx]&0xF0)==0xE0) {
        idx+=3;
      } else {
        return false;
      }
    }

    return true;
  } else if (encoding == GBK) {
    return true;
  } else {
    return false;
  }
  return true;
}

}       //  end for namespace codecs
}       //  end for namespace strutils
}       //  end for namespace ltp
#endif  //  end for __CODECS_HPP__
