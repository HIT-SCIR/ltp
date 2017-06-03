//
// Created by liu on 2016/12/19.
//

#ifndef PROJECT_DEBUG_H
#define PROJECT_DEBUG_H

#include "stdarg.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include "config.h"
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;

namespace base {
  enum LogLevel {
    info = 0, debug, war, err
  };

  class DebugConfig: virtual public base::config {
  public:
    int logLevel;
    string enabledModels;
    DebugConfig(string confName = "Configuration"):base::config(confName) {
      registerConf<int>     ("loglevel"        , INT  , logLevel         , " 0 = err, war, debug, info",0);
      registerConf<string>  ("debugModels"     , STRING  , enabledModels , "debuginfo enabled Models name list","*");
    }
  };

  class Debug {
    static set<string> enabledModels;
    static int logLevel;
    static char* tmpBuffer;
  public:
    string modelName;
    bool disable = true;

    static void init(DebugConfig config);

    Debug(string modelName);

    inline void debug(string msg, ...) {
      va_list ap; va_start(ap, msg); vsprintf(tmpBuffer, msg.c_str(), ap); va_end(ap);
      if (!disable && logLevel <= LogLevel::debug)
        printAtTime(" \x1b[;32mDEBUG\x1b[0m   "  + modelName + "\t \x1b[;32m" +  tmpBuffer + "\x1b[0m");
    }

    inline void info(string msg, ...) {
      va_list ap; va_start(ap, msg); vsprintf(tmpBuffer, msg.c_str(), ap); va_end(ap);
      if (!disable && logLevel <= LogLevel::info)
        printAtTime(" \x1b[;34mINFO\x1b[0m    "  + modelName + "\t \x1b[;34m" +  tmpBuffer + "\x1b[0m");
    }

    inline void warning(string msg, ...) {
      va_list ap; va_start(ap, msg); vsprintf(tmpBuffer, msg.c_str(), ap); va_end(ap);
      if (!disable && logLevel <= LogLevel::war)
        printAtTime(" \x1b[;33mWARNING\x1b[0m " + modelName + "\t \x1b[;33m" +  tmpBuffer + "\x1b[0m");
    }

    inline void error(string msg, ...) {
      va_list ap; va_start(ap, msg); vsprintf(tmpBuffer, msg.c_str(), ap); va_end(ap);
      if (!disable && logLevel <= LogLevel::err)
        printAtTime(" \x1b[;31mERROR\x1b[0m   " + modelName + "\t \x1b[;31m" +  tmpBuffer + "\x1b[0m");
    }

    inline void printAtTime(string str) {
      print('[' + getTime() + ']' + str);
    }

    virtual inline void print(string str) {
      cout << str << endl;
    }

    static inline string getTime(const char daySep = '/', const char secSep = ':') {
      string formatString;
      formatString += "%Y"; formatString += daySep; formatString += "%m"; formatString += daySep;
      formatString += "%d-%H";
      formatString += secSep; formatString += "%M"; formatString += secSep; formatString += "%S";
      time_t t;
      time(&t);
      char buffer [2048];
      strftime (buffer, sizeof(buffer), formatString.c_str(), localtime(&t));
      return string(buffer);
    }
  };
}

#endif //PROJECT_DEBUG_H
