#ifndef __LTP_FRAMEWORK_SERIALIZABLE_H__
#define __LTP_FRAMEWORK_SERIALIZABLE_H__

#include <iostream>

namespace ltp {
namespace framework {

class Serializable {
protected:
  void write_uint(std::ostream & out, unsigned int val) const {
    out.write(reinterpret_cast<const char *>(&val), sizeof(unsigned int));
  }

  unsigned int read_uint(std::istream & in) const {
    char p[4];
    in.read(reinterpret_cast<char*>(p), sizeof(unsigned int));
    return *reinterpret_cast<const unsigned int*>(p);
  }
};

} //  end for framework
} //  end for ltp

#endif  //  end for __LTP_FRAMEWORK_SERIALIZABLE_H__
