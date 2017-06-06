//
// Created by liu on 2017/1/3.
//

#ifndef PROJECT_ABSTRUCTCONVERTER_H
#define PROJECT_ABSTRUCTCONVERTER_H

#include "string"
using namespace std;

namespace extractor {
  template <class T1, class T2>
  class AbstractConverter{
    virtual void init(T1 & origin_data_ref) = 0;
    virtual void run() = 0;
    virtual T2 & getResult() = 0;
    virtual void clear() {};

    static string getClassName() {
      return "Converter<" + T1::getClassName() + ", " + T2::getClassName() + ">";
    }
  };
}

#endif //PROJECT_ABSTRUCTCONVERTER_H
