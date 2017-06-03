//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_MODELTRAINER_H
#define PROJECT_MODELTRAINER_H

#include "../model/Model.h"
#include "AbstractTrainer.h"

namespace model {
  template <class TrainConfigClass>
  class ModelTrainer : public AbstractTrainer<TrainConfigClass> {
  public:
    ModelTrainer(TrainConfigClass & config):AbstractTrainer<TrainConfigClass>(config) {}
    virtual void train() {

    };

    virtual bool checkDev() {
      return false;
    }

  };
}



#endif //PROJECT_MODELTRAINER_H
