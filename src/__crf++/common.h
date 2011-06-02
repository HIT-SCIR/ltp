//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: common.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_COMMON_H__
#define CRFPP_COMMON_H__

#include <setjmp.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef HAVE_CONFIG_H
#ifdef WIN32
#include "config-win32.h"
#else
#include "config.h"
#endif
#endif

#define COPYRIGHT  "CRF++: Yet Another CRF Tool Kit\nCopyright(C)" \
"2005-2007 Taku Kudo, All rights reserved.\n"
#define MODEL_VERSION 100

#if defined(_WIN32) && !defined(__CYGWIN__)
# define OUTPUT_MODE std::ios::binary|std::ios::out
#else
# define OUTPUT_MODE std::ios::out
#endif

#define BUF_SIZE 8192

namespace CRFPP {
  template <class T> inline T _min(T x, T y) { return(x < y) ? x : y; }
  template <class T> inline T _max(T x, T y) { return(x > y) ? x : y; }

  // helper functions defined in the paper
  inline double sigma(double x) {
    if (x > 0) return 1.0;
    else if (x < 0) return -1.0;
    return 0.0;
  }

  template <class Iterator>
  inline size_t tokenizeCSV(char *str,
                            Iterator out, size_t max) {
    char *eos = str + std::strlen(str);
    char *start = 0;
    char *end = 0;
    size_t n = 0;

    for (; str < eos; ++str) {
      while (*str == ' ' || *str == '\t') ++str;  // skip white spaces
      bool inquote = false;
      if (*str == '"') {
        start = ++str;
        end = start;
        for (; str < eos; ++str) {
          if (*str == '"') {
            str++;
            if (*str != '"')
              break;
          }
          *end++ = *str;
        }
        inquote = true;
        str = std::find(str, eos, ',');
      } else {
        start = str;
        str = std::find(str, eos, ',');
        end = str;
      }
      if (max-- > 1) *end = '\0';
      *out++ = start;
      ++n;
      if (max == 0) break;
    }

    return n;
  }

  template <class Iterator>
  inline size_t tokenize(char *str, const char *del,
                         Iterator out, size_t max) {
    char *stre = str + std::strlen(str);
    const char *dele = del + std::strlen(del);
    size_t size = 0;

    while (size < max) {
      char *n = std::find_first_of(str, stre, del, dele);
      *n = '\0';
      *out++ = str;
      ++size;
      if (n == stre) break;
      str = n + 1;
    }

    return size;
  }

  // continus run of space is regarded as one space
  template <class Iterator>
  inline size_t tokenize2(char *str, const char *del,
                          Iterator out, size_t max) {
    char *stre = str + std::strlen(str);
    const char *dele = del + std::strlen(del);
    size_t size = 0;

    while (size < max) {
      char *n = std::find_first_of(str, stre, del, dele);
      *n = '\0';
      if (*str != '\0') {
        *out++ = str;
        ++size;
      }
      if (n == stre) break;
      str = n + 1;
    }

    return size;
  }

  void inline dtoa(double val, char *s) {
    std::sprintf(s, "%-16f", val);
    char *p = s;
    for (; *p != ' '; ++p) {}
    *p = '\0';
    return;
  }

  template <class T> inline void itoa(T val, char *s) {
    char *t;
    T mod;

    if (val < 0) {
      *s++ = '-';
      val = -val;
    }
    t = s;

    while (val) {
      mod = val % 10;
      *t++ = static_cast<char>(mod)+ '0';
      val /= 10;
    }

    if (s == t) *t++ = '0';
    *t = '\0';
    std::reverse(s, t);

    return;
  }

  template <class T>
  inline void uitoa(T val, char *s) {
    char *t;
    T mod;
    t = s;

    while (val) {
      mod = val % 10;
      *t++ = static_cast<char>(mod) + '0';
      val /= 10;
    }

    if (s == t) *t++ = '0';
    *t = '\0';
    std::reverse(s, t);

    return;
  }

#define _ITOA(_n) do { \
char buf[64]; \
itoa(_n, buf); \
append(buf); \
return *this; } while (0)

#define _UITOA(_n) do { \
char buf[64]; \
uitoa(_n, buf); \
append(buf); \
return *this; } while (0)

#define _DTOA(_n) do { \
char buf[64]; \
dtoa(_n, buf); \
append(buf); \
return *this; } while (0)

  class string_buffer: public std::string {
  public:
    string_buffer& operator<<(double _n)             { _DTOA(_n); }
    string_buffer& operator<<(short int _n)          { _ITOA(_n); }
    string_buffer& operator<<(int _n)                { _ITOA(_n); }
    string_buffer& operator<<(long int _n)           { _ITOA(_n); }
    string_buffer& operator<<(unsigned short int _n) { _UITOA(_n); }
    string_buffer& operator<<(unsigned int _n)       { _UITOA(_n); }
    string_buffer& operator<<(unsigned long int _n)  { _UITOA(_n); }
    string_buffer& operator<<(char _n) {
      push_back(_n);
      return *this;
    }
    string_buffer& operator<<(const char* _n) {
      append(_n);
      return *this;
    }
    string_buffer& operator<<(const std::string& _n) {
      append(_n);
      return *this;
    }
  };

  class die {
  public:
    die() {}
    virtual ~die() {
      std::cerr << std::endl;
      exit(-1);
    }
    int operator&(std::ostream&) { return 0; }
  };

  class warn {
  public:
    warn() {}
    virtual ~warn() { std::cerr << std::endl; }
    int operator&(std::ostream&) { return 0; }
  };

  struct whatlog {
    std::ostringstream stream_;
    const char *str() {
      stream_ << std::ends;
      return stream_.str().c_str();
    }
    jmp_buf cond_;
  };

  class wlog {
  public:
    whatlog *l_;
    explicit wlog(whatlog *l): l_(l) { l_->stream_.clear(); }
    virtual ~wlog() { longjmp(l_->cond_, 1); }
    int operator&(std::ostream &) { return 0; }
  };
}

#define WHAT what_.stream_

#define CHECK_RETURN(condition, value) \
if (!(condition)) \
  if (setjmp(what_.cond_) == 1) { \
     return value;  \
  } else \
    wlog(&what_) & what_.stream_ << \
    __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

#define CHECK_0(condition)      CHECK_RETURN(condition, 0)
#define CHECK_FALSE(condition)  CHECK_RETURN(condition, false)

#define CHECK_CLOSE_FALSE(condition) \
if (!(condition)) \
  if (setjmp(what_.cond_) == 1) { \
     close(); \
     return false;  \
  } else \
    wlog(&what_) & what_.stream_ << \
    __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

#define CHECK_DIE(condition) \
(condition) ? 0 : die() & std::cerr << __FILE__ << \
"(" << __LINE__ << ") [" << #condition << "] "

#define CHECK_WARN(condition) \
(condition) ? 0 : warn() & std::cerr << __FILE__ << \
"(" << __LINE__ << ") [" << #condition << "] "
#endif
