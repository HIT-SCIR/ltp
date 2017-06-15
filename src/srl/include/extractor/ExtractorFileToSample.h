//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_ETRACTORFILETOSAMPLE_H
#define PROJECT_ETRACTORFILETOSAMPLE_H

#include "Extractor.h"
#include "../structure/DataFileName.h"
namespace extractor {
  template<class Converter1Class, class Converter2Class, class Converter3Class, class SampleClass>
  class ExtractorFileToSample : public Extractor<DataFileName, SampleClass> {
  public:
    virtual vector<SampleClass> run() {
//        Converter1Class con1;
//        Converter2Class con2;
//        Converter3Class con3;
//        con1.init(*(this->startPtr));
//        con1.run();
//        con2.init(con1.getResult());
//        con2.run();
//        con3.init(con2.getResult());
//        con3.run();
//        return con3.getResult();
      return vector<SampleClass>();
    }
  };
}

#endif //PROJECT_ETRACTORFILETOSAMPLE_H
