#ifndef __LTP_STRUTILS_SBC_DBC_HPP__
#define __LTP_STRUTILS_SBC_DBC_HPP__

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

inline void dbc2sbc(const std::string & x, std::string & y) {
    y = x;
}

}   //  end for namespace chartypes
}   //  end for namespace strutils
}   //  end for namespace ltp
#endif  //  end for __LTP_UTILITY_SBC_DBC_H__
