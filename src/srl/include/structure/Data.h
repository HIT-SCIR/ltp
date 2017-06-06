//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_CORPUSDATA_H
#define PROJECT_CORPUSDATA_H

#include "string"
using namespace std;

namespace extractor {
  class Data {
  public:
    int upDataIndex = 0;
    int upDataInnerIndex = 0;
    static string getClassName() { return "Data"; }
    virtual void clear() {};
  };
}

#endif //PROJECT_CORPUSDATA_H
