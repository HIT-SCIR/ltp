#ifndef __LTP_STRUTILS_SBC_DBC_HPP__
#define __LTP_STRUTILS_SBC_DBC_HPP__

#include "codecs.hpp"
#include "chartypes.hpp"

namespace ltp {
namespace strutils {
namespace chartypes {

/**
 * Convert single byte character into double byte character.
 * [TRICK] to speed up, use y.reserve before calling sbc2dbc.
 * Throughput: 18,000 calls/ms
 *
 *  @param[in]  x   The input single byte character.
 *  @param[out] y   The output double byte character.
 */
inline void sbc2dbc(const std::string& x, std::string& y) {
  unsigned char ch = x[0];
  if ((ch & 0x80) == 0) {
    if (ch>='0' && ch<='9') {
      y = (__chartype_dbc_digit_utf8_buff__[ch-'0']);
    } else if (ch >= 'a' && ch <= 'z') {
      y = (__chartype_dbc_lowercase_utf8_buff__[ch-'a']);
    } else if (ch >= 'A' && ch <= 'Z') {
      y = (__chartype_dbc_uppercase_utf8_buff__[ch-'A']);
    } else if (ch >= 32 && ch <= 47) {
      y = (__chartype_dbc_punc_utf8_buff__[ch-32]);
    } else if (ch >= 58 && ch <= 64) {
      y = (__chartype_dbc_punc_utf8_buff__[48-32+ch-58]);
    } else if (ch >= 91 && ch <= 96) {
      y = (__chartype_dbc_punc_utf8_buff__[48-32+65-58+ch-91]);
    } else if (ch >= 123 && ch <= 126) {
      y = (__chartype_dbc_punc_utf8_buff__[48-32+65-58+97-91+ch-123]);
    } else {
      y = x;
    }
  } else {
    y = x;
  }
}

/**
 * Another version of sbc2dbc, with return the value.
 * Throughput: 10,000 calls/ms
 *
 *  @param[in]  x       The input single byte character.
 *  @return     string  The output double byte character.
 */
inline std::string sbc2dbc(const std::string & x) {
  std::string y; y.reserve(x.size()* 3);
  sbc2dbc(x, y);
  return y;
}

/**
 * Convert a string in single byte character into string in 
 * double byte character. 
 * [TRICK] to speed up, use y.reserve before calling sbc2dbc.
 *
 *  @param[in]  x   The input single byte string.
 *  @param[out] y   The output double byte string.
 */
inline void sbc2dbc_x(const std::string& x, std::string& y,
    int encoding=strutils::codecs::UTF8) {
  int len = x.size();
  int i = 0;
  codecs::iterator itx(x, encoding);
  for (; itx.is_good() && !itx.is_end(); ++ itx) {
    y.append(sbc2dbc(x.substr(itx->first, itx->second- itx->first)));
  }
}

/**
 * Convert a string in single byte character into string in 
 * double byte character and return its word type.
 * [TRICK] to speed up, use y.reserve before calling sbc2dbc.
 *
 *  @param[in]  x   The input single byte string.
 *  @param[out] y   The output double byte string.
 */
inline void sbc2dbc_x_wt(const std::string & x, std::string & y,
    int &wordtype, int encoding=strutils::codecs::UTF8) {
  int len = x.size();
  int i = 0;
  std::string tmp = "";
  bool flag = true;
  int pre = -1,cur = -1;
  y.clear();

  codecs::iterator itx(x, encoding);
  for (; itx.is_good() && !itx.is_end(); ++ itx) {
    y.append(sbc2dbc(x.substr(itx->first, itx->second- itx->first)));

    if (flag) {
      cur = chartype(tmp);
      flag = (pre!=-1 && pre != cur) ? (!flag) : (flag);
      pre = cur;
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

inline std::string sbc2dbc_x_wt(const std::string & x,
    int &wordtype, int encoding=strutils::codecs::UTF8) {
  std::string y;
  sbc2dbc_x_wt(x, y, wordtype, encoding);
  return y;
}

inline void dbc2sbc(const std::string & x, std::string & y) {
  // not implemented.
  y = x;
}

}   //  end for namespace chartypes
}   //  end for namespace strutils
}   //  end for namespace ltp
#endif  //  end for __LTP_UTILITY_SBC_DBC_H__
