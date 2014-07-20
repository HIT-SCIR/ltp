#ifndef __LTP_STRUTILS_SBC_DBC_HPP__
#define __LTP_STRUTILS_SBC_DBC_HPP__

#include "codecs.hpp"
#include "chartypes.hpp"
#include "chartypes.tab"

namespace ltp {
namespace strutils {
namespace chartypes {

inline void sbc2dbc(const std::string & x, std::string & y) {
    unsigned char ch = x[0];
    if ((ch & 0x80) == 0) {
        if (ch>='0' && ch<='9') {
            y = (__chartype_dbc_digit_utf8_buff__ + (ch-'0')*4);
        } else if (ch >= 'a' && ch <= 'z') {
            y = (__chartype_dbc_lowercase_utf8_buff__ + (ch-'a')*4);
        } else if (ch >= 'A' && ch <= 'Z') {
            y = (__chartype_dbc_uppercase_utf8_buff__ + (ch-'A')*4);
        } else if (ch >= 32 && ch <= 47) {
            y = (__chartype_dbc_punc_utf8_buff__ + (ch-32)*4);
        } else if (ch >= 58 && ch <= 64) {
            y = (__chartype_dbc_punc_utf8_buff__ + (48-32)*4+(ch-58)*4);
        } else if (ch >= 91 && ch <= 96) {
            y = (__chartype_dbc_punc_utf8_buff__ + (48-32)*4+(65-58)*4+(ch-91)*4);
        } else if (ch >= 123 && ch <= 126) {
            y = (__chartype_dbc_punc_utf8_buff__ + (48-32)*4+(65-58)*4+(97-91)*4+(ch-123)*4);
        } else {
            y = x;
        }
    } else {
        y = x;
    }
}

inline std::string sbc2dbc(const std::string & x) {
    std::string y; y.reserve(x.size()* 3);
    sbc2dbc(x, y);
    return y;
}

inline void sbc2dbc_x(const std::string & x, std::string & y, int encoding=strutils::codecs::UTF8) {
    int len = x.size();
    int i = 0;

    y.clear();
    while (i<len) {
        if (encoding==strutils::codecs::UTF8) {
            if ((x[i]&0x80)==0) {
                y.append(sbc2dbc(x.substr(i, 1)));
                ++i;
            } else if ((x[i]&0xE0)==0xC0) {
                y.append(sbc2dbc(x.substr(i, 2)));
                i+=2;
            } else if ((x[i]&0xF0)==0xE0) {
                y.append(sbc2dbc(x.substr(i, 3)));
                i+=3;
            } else if ((x[i]&0xF8)==0xF0) {
                y.append(sbc2dbc(x.substr(i, 4)));
                i+=4;
            } else {
                y = x;
                i=len;
            }
        } else if (encoding==strutils::codecs::GBK) {
            // not implemented
        }
    }
}

inline void sbc2dbc_x_wt(const std::string & x, std::string & y, int &wordtype, int encoding=strutils::codecs::UTF8) {
    int len = x.size();
    int i = 0;
    std::string tmp = "";
    bool flag = true;
    int pre = -1,cur = -1;
    y.clear();
    while (i<len) {
        if (encoding==strutils::codecs::UTF8) {
            if ((x[i]&0x80)==0) {
                tmp = sbc2dbc(x.substr(i, 1));
                y.append(tmp);
                ++i;
            } else if ((x[i]&0xE0)==0xC0) {
                tmp = sbc2dbc(x.substr(i, 2));
                y.append(tmp);
                i+=2;
            } else if ((x[i]&0xF0)==0xE0) {
                tmp = sbc2dbc(x.substr(i, 3));
                y.append(tmp);
                i+=3;
            } else if ((x[i]&0xF8)==0xF0) {
                tmp = sbc2dbc(x.substr(i, 4));
                y.append(tmp);
                i+=4;
            } else {
                flag = false;
                y = x;
                i=len;
            }
            if(flag){
                cur = chartype(tmp);
                flag = (pre!=-1 && pre != cur) ? (!flag) : (flag);
                pre = cur;
            }
        } else if (encoding==strutils::codecs::GBK) {
            // not implemented
        }
    }
    if(flag && cur != -1){
        wordtype = cur;
    }
}

inline std::string sbc2dbc_x(const std::string & x, int encoding=strutils::codecs::UTF8) {
    std::string y;
    sbc2dbc_x(x, y, encoding);
    return y;
}

inline std::string sbc2dbc_x_wt(const std::string & x, int &wordtype, int encoding=strutils::codecs::UTF8) {
    std::string y;
    sbc2dbc_x_wt(x, y, wordtype, encoding);
    return y;
}

inline void dbc2sbc(const std::string & x, std::string & y) {
    y = x;
}

}   //  end for namespace chartypes
}   //  end for namespace strutils
}   //  end for namespace ltp
#endif  //  end for __LTP_UTILITY_SBC_DBC_H__
