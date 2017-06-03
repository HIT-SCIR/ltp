//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_CONVERTERCONCEPTTOSAMPLE_H
#define PROJECT_CONVERTERCONCEPTTOSAMPLE_H

#include "../base/debug.h"
#include "BiConverter.h"
#include "../structure/DataSample.h"
#include "../structure/DataConcept.h"
namespace extractor {
  template <class T1 = DataConcept, class T2 = DataSample>
  class ConverterConceptToSample : public BiConverter<T1, T2> {
  public:
    base::Debug debug;
    ConverterConceptToSample():debug("ConverterConceptToSample") {}

    //  由子类实现 virtual void convert(T1 &) = 0;
    //  virtual void convert(T1 &) {
    //    debug.debug("testing ConverterConceptToSample Class - convert foo.");
    //    debug.error("this is not finished yet.");
    //  }
  };
}

#endif //PROJECT_CONVERTERCONCEPTTOSAMPLE_H
