//
// Created by liu on 2017/5/5.
//

#ifndef PROJECT_DYNETTRAINER_H
#define PROJECT_DYNETTRAINER_H

#include "process/ModelTrainer.h"
#include "config/ModelConf.h"
template <class TrainConfigClass>
class DynetTrainer : public model::ModelTrainer<TrainConfigClass> {
  DynetConf& config;

public:
  DynetTrainer(TrainConfigClass & config) :
          model::ModelTrainer<TrainConfigClass>(config),
          config(config)
  { }

  void initDynet() {
    dynet::DynetParams dynetParams;
    dynetParams.mem_descriptor = config.dynet_mem;
    dynetParams.random_seed = config.dynet_seed;
    dynetParams.requested_gpus = config.dynet_gpus;
    dynet::initialize(dynetParams);
  }
  // 工具函数
protected:

  string statusOfSgd(SimpleSGDTrainer & sgd) {
    char s[64];
    sprintf(s, "[epoch=%.2f eta=%.2e clips=%.1f updates=%.0f]", sgd.epoch, sgd.eta, sgd.clips, sgd.updates);
    sgd.updates = sgd.clips = 0;
    return string(s);
  }
};


#endif //PROJECT_DYNETTRAINER_H
