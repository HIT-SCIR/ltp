//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_DATACONCEPT_H
#define PROJECT_DATACONCEPT_H

#include "Data.h"

namespace extractor {
  class DataConcept: public Data {
  public:
    static string getClassName() { return "DataConcept"; }

  };
}

#endif //PROJECT_DATACONCEPT_H
