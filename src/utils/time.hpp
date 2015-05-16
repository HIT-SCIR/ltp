#ifndef __TIME_HPP__
#define __TIME_HPP__

#if _WIN32
// [FROM]! http://stackoverflow.com/questions/1372480/
#define _WINSOCKAPI_    // stops windows.h including winsock.h
#include <windows.h>
#else
#include <sys/time.h>
#endif	//	end for _WIN32

#include <sys/types.h>
#include <ctime>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace ltp {
namespace utility {

class timer {
private:
  std::clock_t _start_time;
public:
  timer() {
    _start_time = std::clock();
  }

  void restart() {
    _start_time = std::clock();
  }

  double elapsed() const {
    return double(std::clock() - _start_time) / CLOCKS_PER_SEC;
  }
};

} //  namespace utility
} //  namespace ltp

#endif  //  end for __TIME_HPP__
