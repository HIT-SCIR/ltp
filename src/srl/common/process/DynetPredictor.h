//
// Created by liu on 2017/5/5.
//

#ifndef PROJECT_DYNETPREDICTOR_H
#define PROJECT_DYNETPREDICTOR_H

#include <dynet/init.h>
#include "process/ModelPredictor.h"
#include "config/ModelConf.h"

template <class PredConfigClass>
class DynetPredictor : public model::ModelPredictor<PredConfigClass> {
  DynetConf& config;

public:
  DynetPredictor(PredConfigClass & config) :
          model::ModelPredictor<PredConfigClass>(config),
          config(config)
  { }

  void initDynet() {
    dynet::DynetParams dynetParams;
    dynetParams.mem_descriptor = config.dynet_mem;
    dynetParams.random_seed = config.dynet_seed;
    dynetParams.requested_gpus = config.dynet_gpus;
    dynet::initialize(dynetParams);
  }
};

#endif //PROJECT_DYNETPREDICTOR_H
