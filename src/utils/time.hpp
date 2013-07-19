#ifndef __TIME_HPP__
#define __TIME_HPP__

#if _WIN32
#include <time.h>
// #include <Windows.h>
#else
#include <sys/time.h>
#endif	//	end for _WIN32

#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
//#include <unistd.h>
#include <stdio.h>
#include <math.h>

namespace ltp {
namespace utility {

inline double get_time(void) {
#if _WIN32
	// not provided
	return -1;
#else
	struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
#endif
}

}
}

#endif  //  end for __TIME_HPP__
