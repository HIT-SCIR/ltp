//
// Created by liu on 2017/1/7.
//

#ifndef PROJECT_ABSTRACTPREDICTOR_H
#define PROJECT_ABSTRACTPREDICTOR_H

#include "../base/process.h"
namespace model {
  template<class TestConfigClass>
  class AbstractPredictor : public base::Process<TestConfigClass>{
  public:
    AbstractPredictor(TestConfigClass &config) : base::Process<TestConfigClass>(config) {}

    virtual void main() {
      init();
      predict();
      extractResult();
    }
    virtual void init() = 0;
    virtual void predict() = 0;
    virtual void extractResult() = 0;

  };
}


#endif //PROJECT_ABSTRACTPREDICTOR_H
