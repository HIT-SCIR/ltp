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

class WallClockTimer {
private:
  double _start_time;
public:
  WallClockTimer() { 
    _start_time = get_time();
  }

  void restart() {
    _start_time = get_time();
  }

  double elapsed() const {
    return get_time() - _start_time;
  }

  static double get_time() {
#if _WIN32
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon  = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min  = wtm.wMinute;
    tm.tm_sec  = wtm.wSecond;
    tm.tm_isdst= -1;
    clock = mktime(&tm);
    double tv_sec = clock;
    double tv_usec = wtm.wMilliseconds * 1000;
    return tv_sec + (tv_usec / 1000000.0);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
#endif
  }
};

} //  namespace utility
} //  namespace ltp

#endif  //  end for __TIME_HPP__
