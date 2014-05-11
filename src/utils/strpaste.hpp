// This serial function is use for optimizing the string concatenation
#ifndef __STRING_PASTE_HPP__
#define __STRING_PASTE_HPP__

#include <iostream>

namespace ltp {
namespace strutils {

inline void paste(std::string &io, 
    const std::string & s1) {
  io.clear();
  io.append(s1);
}

inline void paste(std::string &io, 
    const std::string & s1,
    const std::string & s2) {
  io.clear();
  io.append(s1);
  io.append(s2);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5,
    const std::string & s6) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
  io.append(s6);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5,
    const std::string & s6,
    const std::string & s7) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
  io.append(s6);
  io.append(s7);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5,
    const std::string & s6,
    const std::string & s7,
    const std::string & s8) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
  io.append(s6);
  io.append(s7);
  io.append(s8);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5,
    const std::string & s6,
    const std::string & s7,
    const std::string & s8,
    const std::string & s9) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
  io.append(s6);
  io.append(s7);
  io.append(s8);
  io.append(s9);
}

inline void paste(std::string &io,
    const std::string & s1,
    const std::string & s2,
    const std::string & s3,
    const std::string & s4,
    const std::string & s5,
    const std::string & s6,
    const std::string & s7,
    const std::string & s8,
    const std::string & s9,
    const std::string & s10) {
  io.clear();
  io.append(s1);
  io.append(s2);
  io.append(s3);
  io.append(s4);
  io.append(s5);
  io.append(s6);
  io.append(s7);
  io.append(s8);
  io.append(s9);
  io.append(s10);
}

}       //  end for namespace strutils
}       //  end for namespace ltp

#endif  //  end for __STRING_PASTE_HPP__
