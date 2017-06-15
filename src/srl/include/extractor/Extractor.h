//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_ERACTOR_H
#define PROJECT_ERACTOR_H

#include "AbstractExtractor.h"

#include "vector"
using namespace std;

namespace extractor {
  template<class StartClass, class EndClass>
  class Extractor : public AbstractExtractor<vector<StartClass>, vector<EndClass>>{
  public:
    vector<StartClass> * startPtr;

    virtual void init(vector<StartClass>& start) {
      startPtr = &start;
    };

    virtual vector<EndClass> run() = 0;

    static string getClassName() {
      return "Extractor<" + StartClass::getClassName() + ", " + EndClass::getClassName() + ">";
    }
  };
}

#endif //PROJECT_ERACTOR_H
