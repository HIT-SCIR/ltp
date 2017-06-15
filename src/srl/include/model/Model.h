//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_MODEL_H
#define PROJECT_MODEL_H

#include "../base/debug.h"
#include "string"
using namespace std;

namespace model {
  class Model {
    static string getClassName() { return "Model"; }
    virtual void init() = 0;
    virtual void save() = 0;
    virtual bool load() = 0;
  };
}

#endif //PROJECT_MODEL_H
