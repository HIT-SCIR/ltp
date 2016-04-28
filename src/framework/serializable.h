#ifndef __LTP_FRAMEWORK_SERIALIZABLE_H__
#define __LTP_FRAMEWORK_SERIALIZABLE_H__

#include <iostream>
#include "boost/cstdint.hpp"

namespace ltp {
namespace framework {

using boost::uint32_t;

class Serializable {
protected:
  void write_uint(std::ostream & out, uint32_t val) const {
    out.write(reinterpret_cast<const char *>(&val), sizeof(uint32_t));
  }

  uint32_t read_uint(std::istream & in) const {
    char p[sizeof(uint32_t)];
    in.read(reinterpret_cast<char*>(p), sizeof(uint32_t));
    return *reinterpret_cast<const uint32_t*>(p);
  }
};

} //  end for framework
} //  end for ltp

#endif  //  end for __LTP_FRAMEWORK_SERIALIZABLE_H__
