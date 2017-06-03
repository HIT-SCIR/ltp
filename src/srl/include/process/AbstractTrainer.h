//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_ABSTRACTTRAINER_H
#define PROJECT_ABSTRACTTRAINER_H

#include "../base/process.h"
namespace model {
  template <class TrainConfigClass>
  class AbstractTrainer : public base::Process<TrainConfigClass> {
  public:
    AbstractTrainer(TrainConfigClass & config):base::Process<TrainConfigClass>(config) {}

    virtual void main() {
      this->init();
      this->train();
    }

    virtual void train() = 0;
    virtual void init() {};
  };
}



#endif //PROJECT_ABSTRACTTRAINER_H
