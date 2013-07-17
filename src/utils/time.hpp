#ifndef __TIME_HPP__
#define __TIME_HPP__

#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

namespace ltp {
namespace utility {

inline double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

}
}

#endif  //  end for __TIME_HPP__
