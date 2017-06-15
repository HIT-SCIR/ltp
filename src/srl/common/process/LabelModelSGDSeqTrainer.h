//
// Created by liu on 2017/2/22.
//

#ifndef PROJECT_SGDSeqTRAINER_H
#define PROJECT_SGDSeqTRAINER_H

#include <dynet/training.h>
#include <dynet/init.h>
#include "model/SeqLabelModel.h"
#include "base/debug.h"
#include "config/ModelConf.h"
#include "./DynetTrainer.h"
#include "base/timer.h"
#include "model/LoopCounter.h"
#include "model/RandomOrderMap.h"
#include "process/ConditionStopper.h"
#include "process/TrainStats.h"

using namespace dynet;

template <class TrainConfigClass, class SimpleClass>
class LabelModelSGDSeqTrainer : public DynetTrainer<TrainConfigClass> {
  LabelModelTrainerConf &config;
  base::Debug debug;
public:
  vector<SimpleClass> trainSamples, devSamples;
  SimpleSGDTrainer * sgd;
  Performance bestDevPerf;
  SeqLabelModel<SimpleClass> & labelModel;
  bool useDropOut = false;

  LabelModelSGDSeqTrainer(TrainConfigClass &config, SeqLabelModel<SimpleClass> & labelModel) :
          DynetTrainer<TrainConfigClass>(config), config(config), debug(getClassName()),
          labelModel(labelModel) {
    DynetTrainer<TrainConfigClass>::initDynet();
    resetDropOut();
  }

  virtual void train () {
    debug.debug("Training start");


    sgd = new SimpleSGDTrainer(labelModel.model, config.et0, config.eta_decay);

    unsigned trainSetSize = (unsigned)trainSamples.size();
    unsigned batchSize = min(config.batch_size, trainSetSize);

    checkDev();
    model::LoopCounter trainSetInnerIter(trainSetSize);
    process::ConditionStopper conditionStopper;
    TrainStats trainStats;
    model::RandomOrderMap order((unsigned)trainSamples.size());
    int turn_iter; int lastSaveTurn = 0;
    for (turn_iter = 1; turn_iter <= config.max_iter; turn_iter++) {
      // 迭代一个batch
      trainStats.newBatch();
      Performance perf;
      for (unsigned batchInnerIter = 0; batchInnerIter < batchSize; batchInnerIter++) {
        SimpleClass &samples = trainSamples[order++];
        double err = trainOneSampleGroup(samples, perf);
        if (err < 0.0) labelModel.load();
        trainStats.updateSample(err, samples.size());
      }
      debug.info("%s %s %s",
                 DynetTrainer<TrainConfigClass>::statusOfSgd(*sgd).c_str(),
                 trainStats.getBatchStats().c_str(),
                 perf.toString().c_str());

      sgd->update_epoch((float)batchSize/trainSetSize);

      // check dev set to save
      if (turn_iter % config.batches_to_save == 0 && checkDev()) {
          labelModel.save(); lastSaveTurn = 0;
      } else {
        lastSaveTurn ++;
      }

      // auto stop training
      if (config.use_auto_stop && conditionStopper.auto_end(perf, bestDevPerf, turn_iter, sgd->epoch, lastSaveTurn)) {
        debug.debug("auto finish training.");
        break;
      }
    }
    trainStats.printTrainEndStats();
  }

  virtual bool checkDev() {
    Performance dev_perf;
    unsigned dev_size = (unsigned) devSamples.size();
    disableDropOut();
    base::Timer t;
    for (int j = 0; j < dev_size; ++j) {
      ComputationGraph hg;
      vector<Expression> results = labelModel.label(hg, devSamples[j]);
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
    vector<Expression> results = labelModel.label(hg, sampleGroup);
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
      debug.warning(" got NAN err, sgd reset : %s", DynetTrainer<TrainConfigClass>::statusOfSgd(*sgd).c_str());
      return -1; // for reload model
    }
  }

  static string getClassName() {
    return "LabelModelSGDSeqTrainer";
  }

};


#endif //PROJECT_SGDTRAINER_H
