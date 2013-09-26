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

inline int decode(const std::string & s, 
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
                std::cerr << "Warning: " 
                    << "in utf.h "
                    << "getCharactersFromUTF8String: string '" 
                    << s 
                    << "' not encoded in unicode utf-8" 
                    << std::endl;
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

inline int length(const std::string & s, int encoding=UTF8) {
    unsigned int idx = 0;
    unsigned int len = 0;

    if (encoding == UTF8) {
        while (idx<s.length()) {
            if ((s[idx]&0x80)==0) {
                ++len;
                ++idx;
            } else if ((s[idx]&0xE0)==0xC0) {
                ++len;
                idx+=2;
            } else if ((s[idx]&0xF0)==0xE0) {
                ++len;
                idx+=3;
            } else {
                std::cerr << "Warning: " 
                    << "in utf.h "
                    << "getCharactersFromUTF8String: string '" 
                    << s 
                    << "' not encoded in unicode utf-8" 
                    << std::endl;
                 ++len;
                 ++idx;
            }
        }
    } else if (encoding == GBK) {
        while (idx<s.length()) {
            if ((s[idx]&0x80)==0) {
                ++ len;
                ++ idx;
            } else {
                ++ len;
                idx += 2;
            }
        }
    } else {
        return 0;
    }

    return len;
}

inline bool initial(const std::string & s, 
        std::string & ch, int encoding=UTF8) {
    if (s=="") {
        return false;
    }

    if (encoding == UTF8) {
        if ((s[0]&0x80)==0) {
            ch = s.substr(0, 1);
        } else if ((s[0]&0xE0)==0xC0) {
            ch = s.substr(0, 2);
        } else if ((s[0]&0xF0)==0xE0) {
            ch = s.substr(0, 3);
        } else {
            return false;
        }
    } else {
    }

    return true;
}

inline bool tail(const std::string & s,
        std::string & ch, int encoding=UTF8) {
    int len = s.size();
    if (!len) {
        return false;
    }

    if (encoding=UTF8) {
        if ((s[len-1]&0x80)==0) {
            ch = s.substr(len-1, 1);
        } else if ((len>=2 && (s[len-2]&0xE0)==0xC0)) {
            ch = s.substr(len-2, 2);
        } else if ((len>=3 && (s[len-3]&0xF0)==0xE0)) {
            ch = s.substr(len-3, 3);
        } else {
            return false;
        }
    }
    return false;
}

inline bool isclear(const std::string & s, int encoding=UTF8) {
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
