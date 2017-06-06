//
// Created by liu on 2017/2/22.
//

#ifndef PROJECT_MODELCONF_H
#define PROJECT_MODELCONF_H

#include "base/config.h"
#include "base/debug.h"

class DynetConf : virtual public base::DebugConfig {
public:
  int dynet_gpus;
  string dynet_mem;
  string dynet_gpu_ids;
  unsigned dynet_seed;
  DynetConf(string confName = "Configuration"): base::DebugConfig(confName) {
    registerConf<string>("dynet-mem", STRING, dynet_mem, "", "1000");
    registerConf<unsigned> ("dynet-seed", UNSIGNED, dynet_seed, "dynet_seed", 0);
    registerConf<int>("dynet-gpus", INT, dynet_gpus, "", -1);
    registerConf<string>("dynet-gpu-ids", STRING, dynet_gpu_ids, "", "0");
  }

};

class ModelConf : virtual public DynetConf {
public:

  string model;
  string activate;

  ModelConf(string confName = "Configuration"): DynetConf(confName) {
    registerConf<string>  ("model,m"         , STRING  , model         , "model path"            );
    registerConf<string>  ("activate"           , STRING  , activate     , "activate"              , "rectify");
  }

};


class LabelModelTrainerConf : virtual public ModelConf {
public:

  string training_data;
  string dev_data;
  float et0;
  float eta_decay;
  float best_perf_sensitive;
  unsigned max_iter;
  unsigned batch_size;
  unsigned batches_to_save;

  bool use_dropout;
  float dropout_rate;

  int use_auto_stop;

  LabelModelTrainerConf(string confName = "Configuration"): ModelConf(confName) {
    registerConf<string>  ("training_data,T" , STRING  , training_data , "Training corpus"           );
    registerConf<string>  ("dev_data,d"      , STRING  , dev_data      , "Development corpus"        );
    registerConf<float>   ("learning_rate"   , FLOAT   , et0           , "learning rate"             ,0.1);
    registerConf<float>   ("eta_decay"       , FLOAT   , eta_decay     , "eta_decay"                 ,0.08);
    registerConf<float>   ("best_perf_sensitive", FLOAT, best_perf_sensitive, "min f upgrade to save model",0.00);
    registerConf<unsigned>("max_iter"        , UNSIGNED, max_iter      , "max training iter(batches)",5000);
    registerConf<unsigned>("batch_size"      , UNSIGNED, batch_size    , "batch_size"                ,1000);
    registerConf<unsigned>("batches_to_save" , UNSIGNED, batches_to_save,"after x batches to save model",10);

    registerConf<bool>    ("use_dropout"     , BOOL    , use_dropout   , "Use dropout"               );
    registerConf<float>   ("dropout_rate"    , FLOAT   , dropout_rate  , "dropout rate"              ,0.5);

    registerConf<int>    ("use_auto_stop"   , INT    , use_auto_stop   , "Use auto stop"            , 0);
  }
};

class LabelModelPredictorConf : virtual public ModelConf {
public:
  string test_data;
  string output;
  LabelModelPredictorConf(string confName = "Configuration"): ModelConf(confName) {
    registerConf<string>  ("test_data,p"     , STRING  , test_data     , "Test corpus"                );
    registerConf<string>  ("output,o"        , STRING  , output        , "Testing output labels"      );
  }
};

#endif //PROJECT_MODELCONF_H
