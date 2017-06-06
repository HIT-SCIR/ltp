//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_PROCESS_H
#define PROJECT_PROCESS_H

#include "config.h"
namespace base {
  template <class ConfigClass>
  class Process {
  public:
    ConfigClass & config;
    int returnNum = 0;

    Process(ConfigClass &config): config(config) {}

    virtual void main() = 0;

    static ConfigClass createConfig() {
      return ConfigClass();
    }
  };
}



#endif //PROJECT_PROCESS_H
