//
// Created by liu on 2017/2/22.
//

#ifndef PROJECT_BILSTMBASEMODEL_H
#define PROJECT_BILSTMBASEMODEL_H

#include <base/debug.h>
#include <dynet/model.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include "structure/Performance.h"
#include "structure/Prediction.h"
#include "config/ModelConf.h"
#include "Const.h"
#include "./BaseLabelModel.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace dynet;
using namespace dynet::expr;

template <class SampleClass>
class SeqLabelModel : public BaseLabelModel<SampleClass>  {
  ModelConf& config;
  base::Debug debug;
public:
  float dropout_rate = 0; // 默认关闭dropout

  SeqLabelModel(ModelConf &config) :
          BaseLabelModel<SampleClass>(config),
          config(config), debug("SeqLabelModel") { }

  virtual vector<Expression> label(ComputationGraph& hg, SampleClass & samples) = 0;

  virtual Expression ExtractError(ComputationGraph& hg, vector<Expression>& adists, SampleClass & samples, Performance &perf) = 0;

  virtual void ExtractResults(ComputationGraph &hg, vector<Expression> &adists, SampleClass &samples) = 0;

};


#endif //PROJECT_BILSTMBASEMODEL_H
