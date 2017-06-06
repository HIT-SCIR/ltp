//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_CONVERTERBLOCKTOCONCEPT_H
#define PROJECT_CONVERTERBLOCKTOCONCEPT_H

#include "BiConverter.h"
#include "../structure/DataFileBlockContext.h"
#include "../structure/DataConcept.h"
#include "../base/debug.h"


namespace extractor {
  template <class T1 = DataFileBlockContext, class T2 = DataConcept>
  class ConverterBlockToConcept: public BiConverter<T1, T2> {
  public:
    base::Debug debug;
    ConverterBlockToConcept():debug("ConverterBlockToConcept") {}

    virtual void convert(T1 & fileBlock) {
      debug.debug("testing ConverterBlockToConcept Class - convert foo.");
    }
  };
}


#endif //PROJECT_CONVERTERBLOCKTOCONCEPT_H
