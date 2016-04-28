#ifndef __LTP_UTILS_STRUTILS_HPP__
#define __LTP_UTILS_STRUTILS_HPP__

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

namespace ltp {
namespace strutils {

inline void trim(std::string& str) {
  size_t len = str.size();
  if (len == 0) { return; }

  while (len >= 1 && (str[len-1] == ' '
        || str[len-1]=='\t'
        || str[len-1] == '\r'
        || str[len-1] == '\n')) {
    -- len;
  }
  str = str.substr(0, len);

  size_t i = 0;
  while (i < len && (str[i] == ' ' ||
         str[i] == '\t' ||
         str[i] == '\r' ||
         str[i] == '\n')) { i ++; }
  str = str.substr(i);
}

/**
 * chomp a string
 *
 *  @param  str     std::string
 *  @return         std::string
 */
inline std::string trim_copy(const std::string& source) {
  std::string str(source);
  trim(str);
  return str;
}

/**
 * Cut off the following string after mark
 *
 *  @param  str     std::string     the string
 *  @param  mark    std::string     the cut out mark
 *  @return         std::string     the cut string
 */
inline std::string cutoff(std::string str, std::string mark) {
  size_t pos = str.find(mark);
  if (pos == std::string::npos) {
    return str;
  } else {
    return str.substr(0, pos);
  }
}

inline void split(const std::string& source, std::vector<std::string>& ret,
    int maxsplit=-1) {
  std::string str(source);
  int numsplit = 0;
  int len = str.size();
  size_t pos;
  for (pos = 0; pos < str.size() && (str[pos] == ' ' || str[pos] == '\t'
        || str[pos] == '\n' || str[pos] == '\r'); ++ pos);
  str = str.substr(pos);

  ret.clear();
  while (str.size() > 0) {
    pos = std::string::npos;

    for (pos = 0; pos < str.size() && (str[pos] != ' '
          && str[pos] != '\t'
          && str[pos] != '\r'
          && str[pos] != '\n'); ++ pos);

    if (pos == str.size()) {
      pos = std::string::npos;
    }

    if (maxsplit >= 0 && numsplit < maxsplit) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    } else if (maxsplit >= 0 && numsplit == maxsplit) {
      ret.push_back(str);
      ++ numsplit;
    } else if (maxsplit == -1) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    }

    if (pos == std::string::npos) {
      str = "";
    } else {
      for (; pos < str.size() && (str[pos] == ' '
            || str[pos] == '\t'
            || str[pos] == '\n'
            || str[pos] == '\r'); ++ pos);
      str = str.substr(pos);
    }
  }
}

/**
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> split(const std::string& source, int maxsplit = -1) {
  std::vector<std::string> ret;
  split(source, ret, maxsplit);
  return ret;
}

/**
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  sep         std::string     the separator
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> split_by_sep(std::string str,
    std::string sep = "", int maxsplit = -1) {
  std::vector<std::string> ret;

  int numsplit = 0;
  int len      = str.size();
  int sep_flag = (sep != "");

  if (sep == "") {
    return split(str, maxsplit);
  }

  while (str.size() > 0) {
    size_t pos = std::string::npos;
    pos = str.find(sep);

    if (maxsplit >= 0 && numsplit < maxsplit) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    } else if (maxsplit >= 0 && numsplit == maxsplit) {
      ret.push_back(str);
      pos = std::string::npos;
      ++ numsplit;
    } else if (maxsplit == -1) {
      ret.push_back(str.substr(0, pos));
      ++ numsplit;
    }

    if (pos == std::string::npos) {
      str = "";
    } else {
      pos = pos + sep.size();
      str = str.substr(pos);
    }
  }
  return ret;
}


/**
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> rsplit(std::string str, int maxsplit = -1) {
  std::vector<std::string> ret;

  int numsplit = 0;
  int len = -1;

  while ((len = str.size()) > 0) {
    // warning: should not use size_t, because it's unsigned integer
    int pos = 0;
    for (pos = len - 1; pos >= 0 && (str[pos] != ' '
          && str[pos] != '\t'
          && str[pos] != '\r'
          && str[pos] != '\n'); -- pos);

    if (maxsplit >= 0 && numsplit < maxsplit) {
      ret.push_back(str.substr(pos + 1));
      ++ numsplit;
    } else if (maxsplit >= 0 && numsplit == maxsplit) {
      ret.push_back(str);
      pos = -1;
      ++ numsplit;
    } else if (maxsplit == -1) {
      ret.push_back(str.substr(pos + 1));
      ++ numsplit;
    }

    if (pos == -1) {
      str = "";
    } else {
      for (; pos >= 0 && (str[pos] == ' '
            || str[pos] == '\t'
            || str[pos] == '\n'
            || str[pos] == '\r'); -- pos);
      str = str.substr(0, pos + 1);
    }
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

/**
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  sep         std::string     the separator
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */

inline std::vector<std::string> rsplit_by_sep(std::string str, std::string sep = "", int maxsplit = -1) {
    std::vector<std::string> ret;

    int numsplit = 0;
    int len      = str.size();
    int sep_flag = (sep != "");

    while (str.size() > 0) {
        int pos = (int) std::string::npos;

        if (sep_flag) {
            pos = str.rfind(sep);
        } else {
            for (pos = str.size() - 1; pos >= 0; -- pos) {
                if (str[pos] == ' '
                        || str[pos] == '\t'
                        || str[pos] == '\r'
                        || str[pos] == '\n') {
                    break;
                }
            }
            if (pos == -1) {
                pos = 0;
                // pos = std::string::npos;
            }
        }

        if (maxsplit >= 0 && numsplit < maxsplit) {
            ret.push_back(str.substr(pos + sep.length(), std::string::npos));
            ++ numsplit;
        } else if (maxsplit >= 0 && numsplit == maxsplit) {
            ret.push_back(str);
            pos = 0;
            ++ numsplit;
        } else if (maxsplit == -1) {
            ret.push_back(str.substr(pos, std::string::npos));
            ++ numsplit;
        }

        if (pos == 0) {
            str = "";
        } else {
            if (sep_flag) {
                // pos = pos - sep.size();
            } else {
                for (; pos >= 0 && (str[pos] == ' '
                            || str[pos] == '\t'
                            || str[pos] == '\n'
                            || str[pos] == '\r'); -- pos);
            }
            str = str.substr(0, pos);
        }
    }

    std::reverse(ret.begin(), ret.end());
    return ret;
}

/**
 * Concatenate a list of words
 *
 *  @param  sep std::vector<std::string>  the words
 *  @return     std::string               the concatenated string
 */
inline std::string join(const std::vector<std::string>& sep) {
  std::string ret = "";
  for (unsigned int i = 0; i < sep.size(); ++ i) {
    // trick, append is faster than plus operator
    ret.append(sep[i]);
  }
  return ret;
}


/**
 * Concatenate a list of words invertening the sep
 *
 *  @param  sep       std::vector<std::string>    the words
 *  @param  delimiter std::string                 the delimiter
 *  @return           std::string                 the concatenated string
 */
inline std::string join(const std::vector<std::string>& sep,
    const std::string& delimiter) {
  if (sep.size() == 0) {
    return "";
  }
  std::string ret = sep[0];
  for (unsigned int i = 1; i < sep.size(); ++ i) {
    ret.append(delimiter);
    ret.append(sep[i]);
  }
  return ret;
}

/**
 * Return True if string starts with the prefix, otherwise return False
 *
 *  @param  str   const std::string&  the string
 *  @param  head  const std::string&  the prefix
 *  @return       bool                true if startswith prefix, otherwise false
 */
inline bool startswith(const std::string &str, const std::string &head) {
  int len = head.size();
  return (str.substr(0, len) == head);
}

/**
 * Return True if string ends with the suffix, otherwise return False
 *
 *  @param  str     const std::string&  the string
 *  @param  suffix  const std::string&  the suffix
 *  @return         bool                true if endswith suffix, otherwise false
 */
inline bool endswith(const std::string &str, const std::string &suffix) {
  int len = suffix.length();
  int len2 = str.length();
  if (len2 < len) {
    return false;
  }

  return (str.substr(len2 - len, len) == suffix);
}


/**
 * Return True if string is integer
 *
 *  @param  str     const std::string&      the string
 *  @return         bool                    true if the string is integer,
 *                                          otherwise false
 */
inline bool is_int(const std::string &str) {
  unsigned int i = 0;
  if (str[0] == '-')
    i = 1;

  for (; i < str.size(); ++ i) {
    if (false == (str[i] >= '0' && str[i] <= '9')) {
      return false;
    }
  }
  return true;
}

/**
 * Return True if string is double
 *
 *  @param  str     const std::string&      the string
 *  @return         bool                    true if the string is double,
 *                                          otherwise false
 */
inline bool is_double(const std::string &str) {
  unsigned int i = 0;
  int state = 0;
  if (str[0] == '-')
    i = 1;

  for (; i < str.size(); ++ i) {
    if (str[i] == '.') {
      ++ state;
      if (state > 1) return false;
    } else if (false == (str[i] >= '0' && str[i] <= '9')) {
      return false;
    }
  }
  return true;
}


/**
 * Convert a string to a plain integer
 *
 *  @param  str     const std::string&      the string
 *  @return         int                     the integer.
 */
inline int to_int(const std::string &str) {
  int ret = 0;
  int sign = 1;
  unsigned int i = 0;
  if ('-' == str[0]) {
    sign = -1;
    i = 1;
  }

  for (; i < str.size(); ++ i) {
    ret *= 10;
    ret += str[i] - '0';
  }
  return sign * ret;
}

/**
 * Convert a string to a double float
 *
 *  @param  str     const std::string&      the string
 *  @return         double                  the double float.
 */
inline double to_double(const std::string &str) {
  double x = 0.0, y = 1.0;
  double sign = 1.0;
  unsigned int i = 0;

  if ('-' == str[0]) {
    sign = -1.0;
    i = 1;
  }

  for (; i < str.size() && str[i] != '.'; ++ i) {
    x *= 10.0;
    x += (str[i] - '0');
  }

  for (++ i; i < str.size(); ++ i) {
    y /= 10.0;
    x += (str[i] - '0') * y;
  }

  return x * sign;
}

inline std::string to_str(int x) {
  char buff[14];
  return std::string(buff, sprintf(buff, "%d", x));
}

// remove the leading space and ending \r\n\t
inline void clean(std::string &str) {
  std::string blank = " \t\r\n";

  size_t pos1 = str.find_first_not_of(blank);
  size_t pos2 = str.find_last_not_of(blank);

  if (pos1 == std::string::npos) {
    str = "";
  } else {
    str = str.substr(pos1, pos2 - pos1 + 1);
  }
}

inline size_t count(const std::string& str, const std::string& sub) {
  if (sub.length() == 0) return 0;
  size_t retval = 0;
  for (size_t offset = str.find(sub); offset != std::string::npos;
      offset = str.find(sub, offset + sub.length())) {
    ++ retval;
  }
  return retval;
}

/**
 *
 *
 *
 *
 */
//int char_type(const std::string &str);

} //LTP_STRING_NAMESPACE_END

} //LTP_NAMESPACE_END

#endif  // end for __LTP_UTILS_STRUTILS_HPP__
