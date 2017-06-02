//
// Created by liu on 2017/2/24.
//

#ifndef PROJECT_LABELMODELSGDPREDICTOR_H
#define PROJECT_LABELMODELSGDPREDICTOR_H

#include <base/progressBar.h>
#include "process/DynetPredictor.h"
#include "model/SeqLabelModel.h"
#include "config/ModelConf.h"
#include "base/timer.h"

template <class PredConfigClass, class SimpleClass>
class LabelModelSGDSeqPredictor : public DynetPredictor<PredConfigClass>{
public:
  LabelModelPredictorConf &config;
  vector<SimpleClass> testSamples;
  SeqLabelModel<SimpleClass> & labelModel;
  base::Debug debug;

  /**
   * @param config
   * @param labelModel 这是向父类传递的模型句柄，至少要实现BaseLabelModel定义的接口
   */
  LabelModelSGDSeqPredictor(PredConfigClass &config, SeqLabelModel<SimpleClass> & labelModel) :
    DynetPredictor<PredConfigClass>(config), config(config),
    labelModel(labelModel),
    debug(getClassName())
  {
    DynetPredictor<PredConfigClass>::initDynet();
  }

  virtual void predict() {
    debug.debug("prediction start.");
    base::ProgressBar bar((int)testSamples.size(), 25);
    base::Timer timer;
    for (int j = 0; j < testSamples.size(); ++j) {
      ComputationGraph hg;
      vector<Expression> adists = labelModel.label(hg, testSamples[j]);
      labelModel.ExtractResults(hg, adists, testSamples[j]);
      if (bar.updateLength(j + 1))
        debug.info("(%d)%s is predicted", j + 1, bar.getProgress(j + 1).c_str());
    }
    debug.debug(" predict %d in %s", (int)testSamples.size(), timer.end().c_str());
  }

  static string getClassName() {
    return "LabelModelSGDPredictor";
  }
};


#endif //PROJECT_LABELMODELSGDPREDICTOR_H
