#ifndef DYNET_HELPER_H_
#define DYNET_HELPER_H_

#include <string>

/// helper functions

namespace dynet {

/**
    this fix a compilation problem in cygwin
*/
#if defined(__CYGWIN__)
  template <typename T>
    inline std::string to_string(T value)
    {
      std::ostringstream os;
      os << value;
      return os.str();
    }
#endif

} // namespace dynet

#endif
