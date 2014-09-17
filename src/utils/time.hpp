#ifndef __TIME_HPP__
#define __TIME_HPP__

#if _WIN32
  #include <time.h>
  static const unsigned __int64 epoch = ((unsigned __int64) 116444736000000000ULL);
#else
  #include <sys/time.h>
#endif	//	end for _WIN32

#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace ltp {
namespace utility {

inline double get_time(void) {
#if _WIN32
  // From: http://stackoverflow.com/questions/1676036/
  FILETIME        file_time;
  SYSTEMTIME      system_time;
  ULARGE_INTEGER  ularge;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  ularge.LowPart = file_time.dwLowDateTime;
  ularge.HighPart = file_time.dwHighDateTime;

  double tv_sec = (long) ((ularge.QuadPart - epoch) / 10000000L);
  double tv_usec = (long) (system_time.wMilliseconds * 1000);

  return tv_sec + (tv_usec / 1000000.0);
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (tv.tv_usec / 1000000.0);
#endif
}

}
}

#endif  //  end for __TIME_HPP__
