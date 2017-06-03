//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_MODELPREDICTOR_H
#define PROJECT_MODELPREDICTOR_H

#include "process/AbstractPredictor.h"
namespace model {
  template<class TestConfigClass>
  class ModelPredictor : public AbstractPredictor<TestConfigClass>{
  public:
    ModelPredictor(TestConfigClass &config) : AbstractPredictor<TestConfigClass>(config) {}

    virtual void init() {

    }

    virtual void predict() {

    }

    virtual void extractResult() {

    }

  };
}



#endif //PROJECT_MODELPREDICTOR_H
