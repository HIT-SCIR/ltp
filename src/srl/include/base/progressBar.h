//
// Created by liu on 2017/1/3.
//

#ifndef PROJECT_PROGRESSBAR_H
#define PROJECT_PROGRESSBAR_H

#include <stdio.h>
#include "iostream"
#include "string"
using namespace std;

namespace base {
  class ProgressBar {
  public:
    int total;
    int currentLength = 0;
    double duration;
    clock_t start;
    ProgressBar(int total, float duration = 1):
            total(total), duration(duration) {
      start = clock();
    }

    inline bool updateLength(int length) {
      currentLength = length;
      clock_t now = clock();
      if (double(now - start)/CLOCKS_PER_SEC > duration) {
        start = now;
        return true;
      }
      return false;
    }

    inline string getProgress(int length) {
      char tmpBuffer[20];
      sprintf(tmpBuffer, "%.2lf%%", 100.0 * length / total);
      return string(tmpBuffer);
    }

    inline string getProgress() {
      return getProgress(currentLength);
    }
  };
}

#endif //PROJECT_PROGRESSBAR_H
