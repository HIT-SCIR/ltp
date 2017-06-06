//
// Created by liu on 2017/5/5.
//

#ifndef PROJECT_LABELMODEL_H
#define PROJECT_LABELMODEL_H

#include "BaseLabelModel.h"
#include "Const.h"

using namespace dynet;
using namespace dynet::expr;

template <class SampleClass>
class LabelModel : public BaseLabelModel<SampleClass>  {
  ModelConf& config;
  base::Debug debug;
public:
  float dropout_rate = 0; // 默认关闭dropout

  LabelModel(ModelConf &config) :
          BaseLabelModel<SampleClass> (config),
          config(config), debug("LabelModel") { }

  virtual Expression label(ComputationGraph& hg, SampleClass & samples) = 0;

  virtual Expression ExtractError(ComputationGraph& hg, Expression& adists, SampleClass & samples, Performance &perf) = 0;
  /**
 * 提取Prediction
 * @param hg
 * @param adists
 * @param answerTable
 * @return
 */
  virtual Prediction ExtractResults(ComputationGraph& hg, Expression& adists) {
    return extractPrediction(as_vector(hg.incremental_forward(adists)));
  }
};

#endif //PROJECT_LABELMODEL_H
