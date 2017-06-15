//
// Created by liu on 2017/2/22.
//

#ifndef PROJECT_SGDTRAINER_H
#define PROJECT_SGDTRAINER_H

#include "process/DynetTrainer.h"
#include "model/BaseLabelModel.h"
#include "base/debug.h"
#include "config/ModelConf.h"
#include "base/timer.h"
#include "dynet/training.h"
#include "model/LoopCounter.h"
#include "model/RandomOrderMap.h"


template <class TrainConfigClass, class SimpleClass>
class LabelModelSGDTrainer : public DynetTrainer<TrainConfigClass> {
public:
  LabelModelTrainerConf &config;
  vector<SimpleClass> trainSamples, devSamples;
  dynet::SimpleSGDTrainer * sgd;
  BaseLabelModel<SimpleClass> & labelModel;
  Performance bestDevPerf;
  base::Debug debug;

  bool useDropOut = false;
  bool enableFirstDevCheck = true;

  LabelModelSGDTrainer(TrainConfigClass &config, BaseLabelModel<SimpleClass> & labelModel) :
          DynetTrainer<TrainConfigClass>(config), config(config),
          labelModel(labelModel),
          debug(getClassName()){
    DynetTrainer<TrainConfigClass>::initDynet();
    resetDropOut();
  }

  virtual void train () {
    sgd = new SimpleSGDTrainer(labelModel.model, config.et0, config.eta_decay);
    debug.debug("Training start");
    base::Timer t;

    unsigned trainSetSize = (unsigned)trainSamples.size();
    unsigned batchSize = min(config.batch_size, trainSetSize);

    model::LoopCounter trainSetInnerIter(trainSetSize);
    double totalSeenSampleNum = 0;
    if (enableFirstDevCheck) checkDev();
    model::RandomOrderMap order((unsigned)trainSamples.size());

    for (int turn_iter = 1; turn_iter <= config.max_iter; turn_iter++) {
      // 迭代一个batch
      Performance perf; // 每个batch 累计统计
      double llh = 0; // 每个batch 的累计lost
      double batchSampleNum = 0; // batch 中词总数
      for (unsigned batchInnerIter = 0; batchInnerIter < batchSize; batchInnerIter++) {
        SimpleClass &samples = trainSamples[order++];
        batchSampleNum ++;
        double err = trainOneSampleGroup(samples, perf);
        if (err < 0.0) labelModel.load();
        llh += err;
        totalSeenSampleNum += 1;
      }
      debug.info("%s update #%d \terr:%.2lf e/b:%lf %s",
                 statusOfSgd(*sgd).c_str(), turn_iter,
                 totalSeenSampleNum/trainSetSize, llh, (llh/batchSampleNum), perf.toString().c_str());
      sgd->update_epoch((float)batchSize/trainSetSize);

      if (turn_iter % config.batches_to_save == 0) {
        if (checkDev()) {
          labelModel.save();
        }
      }
    }
    debug.debug("Training end. Total using %s, iter %u batch, %.0ld samples",
                t.end().c_str(), config.max_iter, totalSeenSampleNum);
  }

  virtual bool checkDev() {
    Performance dev_perf;
    unsigned dev_size = (unsigned) devSamples.size();
    disableDropOut();
    base::Timer t;
    for (int j = 0; j < dev_size; ++j) {
      ComputationGraph hg;
      Expression results = labelModel.label(hg, devSamples[j]);
      labelModel.ExtractError(hg, results, devSamples[j], dev_perf);
    }
    debug.debug(" **dev %s (best f=%lf) [%u samples in %s]", dev_perf.toString().c_str(), bestDevPerf.fscore(), dev_size, t.end().c_str());
    resetDropOut();
    if (dev_perf.fscore() > (bestDevPerf.fscore() + config.best_perf_sensitive)) {
      bestDevPerf = dev_perf;
      return true;
    } else if (dev_perf.fscore() > bestDevPerf.fscore()){
      debug.debug(" this test dev f:%lf is no larger than best %lf %f. The small upgrade will be ignored.",
                  dev_perf.fscore(), bestDevPerf.fscore(), config.best_perf_sensitive);
    }
    return false;
  }

  void resetDropOut() {
    labelModel.setDropOut(config.use_dropout ? config.dropout_rate : 0);
  }

  void disableDropOut() {
    labelModel.setDropOut(0);
  }

protected:

  virtual double trainOneSampleGroup(SimpleClass & sampleGroup, Performance & perf) {
    ComputationGraph hg;
    Expression results = labelModel.label(hg, sampleGroup);
    Expression err = labelModel.ExtractError(hg, results, sampleGroup, perf);
    double lp = as_scalar(hg.incremental_forward(err));
    if (lp >= 0.0) {
      // could feedback err
      hg.backward(err);
      sgd->update(1.0);
      return lp;
    } else {
      // err=nan leaning_rate -= decay
      sgd->update_epoch(0.1);
      debug.warning(" got NAN err, sgd reset : %s", statusOfSgd(*sgd).c_str());
      return -1; // for reload model
    }
  }

  static string getClassName() {
    return "LabelModelSGDTrainer";
  }

};


#endif //PROJECT_SGDTRAINER_H
