//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_ABSTRACTEXTRACTOR_H
#define PROJECT_ABSTRACTEXTRACTOR_H

#include "string"
using namespace std;

namespace extractor {
  template<class StartClass, class EndClass>
  class AbstractExtractor {

    virtual void init(StartClass & start) = 0;

    virtual EndClass run() = 0;

    static string getClassName() {
      return "AbstractExtractor";
    }
  };
}

#endif //PROJECT_ABSTRACTEXTRACTOR_H
