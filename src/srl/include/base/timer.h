//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_TIMER_H
#define PROJECT_TIMER_H

#include <stdio.h>
#include <chrono>
#include "string"
#include "iostream"
using namespace std;

namespace base {
  class Timer {
    chrono::time_point<chrono::high_resolution_clock> startTime, endTime;
  public:
    Timer(): startTime(chrono::high_resolution_clock::now()) { }
    Timer & start() {
      startTime = chrono::high_resolution_clock::now();
      return *this;
    }

    void pause() {
      throw "not implement yet.";
    }
    void resume() {
      throw "not implement yet.";
    }

    string end() {
      endTime = chrono::high_resolution_clock::now();
      return getDuration();
    }

    string getDuration() {
      long long int durationTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
      char buf[50];
      if (durationTime < 1000) {
        sprintf(buf, "%lld us", durationTime);
      } else if (durationTime < 1000l * 1000) {
        sprintf(buf, "%.2lf ms", durationTime/1000.0);
      } else if (durationTime < 1000l * 1000 * 1000) {
        sprintf(buf, "%.2lf s", durationTime/1000.0/1000.0);
      } else if (durationTime < 1000l * 1000 * 1000 * 60) {
        sprintf(buf, "%.2lf min", durationTime/1000.0/1000.0/60.0);
      } else {
        sprintf(buf, "%.2lf h", durationTime/1000.0/1000.0/60.0/60.0);
      }
      return buf;
    }

    static string getClassName() { return "Timer"; }

  };
}

#endif //PROJECT_TIMER_H
