//
// Created by liu on 2017/1/10.
//

#ifndef PROJECT_MODELTESTER_H
#define PROJECT_MODELTESTER_H

#include "AbstractPredictor.h"

namespace model {
  template<class TestConfigClass>
  class ModelTester : public AbstractPredictor<TestConfigClass> {
  public:
    ModelTester(TestConfigClass &config) : AbstractPredictor(config) {}

    virtual void init() {

    }

    virtual void predict() {

    }

    virtual void extractResult() {

    }
  };

}

#endif //PROJECT_MODELTESTER_H
