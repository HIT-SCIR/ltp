/*
 *  libutility - A collection of C/C++ libraries for text processing, 
 *  argument parsing, logging and some other thing.
 *
 *  Copyright (C) 2012-2012 Yijia Liu
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef __STRLIB_X_H__
#define __STRLIB_X_H__

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

namespace ltp { //LTP_NAMESPACE_BEGIN
namespace strutils { //LTP_STRING_NAMESPACE_BEGIN

/*
 * chomp a string
 *
 *  @param  str     std::string
 *  @return         std::string
 */
inline std::string chomp(std::string str) {
    int len = str.size();

    if (len == 0) {
        return "";
    }

    while (str[len-1] == ' ' ||
            str[len-1]=='\t' ||
            str[len-1] == '\r' ||
            str[len-1] == '\n' ) {
        str = str.substr(0, -- len);
    }

    while (str[0] == ' ' ||
            str[0] == '\t' ||
            str[0] == '\r' ||
            str[0] == '\n') {
        str = str.substr(1, -- len);
    }
    return str;
}


/*
 * I don't know why I write this function
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

/*
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  sep         std::string     the separator
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> split(std::string str, int maxsplit = -1) {
    std::vector<std::string> ret;

    int numsplit = 0;
    int len = str.size();

    while (str.size() > 0) {
        size_t pos = std::string::npos;

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

    return ret;
}


/*
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  sep         std::string     the separator
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> split_by_sep(std::string str, std::string sep = "", int maxsplit = -1) {
    std::vector<std::string> ret;

    int numsplit = 0;
    int len      = str.size();
    int sep_flag = (sep != "");

    while (str.size() > 0) {
        size_t pos = std::string::npos;

        if (sep_flag) {
            pos = str.find(sep);
        } else {
            for (pos = 0; pos < str.size(); ++ pos) {
                if (str[pos] == ' '
                        || str[pos] == '\t'
                        || str[pos] == '\r'
                        || str[pos] == '\n') {
                    break;
                }
            }
            if (pos == str.size()) {
                pos = std::string::npos;
            }
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
            if (sep_flag) {
                pos = pos + sep.size();
            } else {
                for (; pos < str.size() && (str[pos] == ' '
                            || str[pos] == '\t'
                            || str[pos] == '\n'
                            || str[pos] == '\r'); ++ pos);
            }
            str = str.substr(pos);
        }
    }

    return ret;
}


/*
 * Return a list of words of string str, the word are separated by
 * separator.
 *
 *  @param  str         std::string     the string
 *  @param  sep         std::string     the separator
 *  @param  maxsplit    std::string     the sep upperbound
 *  @return             std::vector<std::string> the words
 */
inline std::vector<std::string> rsplit(std::string str, int maxsplit = -1) {
    std::vector<std::string> ret;

    int numsplit = 0;
    int len = str.size();

    while (str.size() > 0) {
        size_t pos = std::string::npos;

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
    return ret;
}

/*
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

/*
 * Concatenate a list of words
 *
 *  @param  words   std::vector<std::string>    the words
 *  @return         std::string     the concatenated string
 */
inline std::string join(std::vector<std::string> sep) {
    std::string ret = "";
    for (unsigned int i = 0; i < sep.size(); ++ i) {
        ret = ret + sep[i];
    }
    return ret;
}


/*
 * Concatenate a list of words invertening the sep
 *
 *  @param  words   std::vector<std::string>    the words
 *  @param  sep     std::string                 the sep
 *  @return         std::string     the concatenated string
 */
inline std::string join(std::vector<std::string> sep, std::string separator) {
    if (sep.size() == 0) return "";
    std::string ret = sep[0];
    for (unsigned int i = 1; i < sep.size(); ++ i) {
        ret = ret + separator + sep[i];
    }
    return ret;
}

/*
 * Return True if string starts with the prefix, otherwise return False
 *
 *  @param  str     const std::string&      the string
 *  @param  perfix  const std::string&      the prefix
 *  @return         bool                    true if startswith prefix,
 *                                          otherwise false
 */
inline bool startswith(const std::string &str, const std::string &head) {
    int len = head.size();

    return (str.substr(0, len) == head);
}

inline bool endswith(const std::string &str, const std::string &suffix) {
    int len = suffix.length();
    int len2 = str.length();
    if (len2 < len) {
        return false;
    }

    return (str.substr(len2 - len, len) == suffix);
}


/*
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

/*
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

/*
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

/*
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


/*
 *
 *
 *
 *
 */
//int char_type(const std::string &str);

} //LTP_STRING_NAMESPACE_END
} //LTP_NAMESPACE_END

#endif  // end for __STRLIB_X_H__
