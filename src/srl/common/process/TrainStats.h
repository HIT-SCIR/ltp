//
// Created by liu on 2017/5/11.
//

#ifndef PROJECT_TRAINSTATS_H
#define PROJECT_TRAINSTATS_H

#include "base/debug.h"
#include "base/timer.h"

class TrainStats {
  base::Debug debug;
  // global
  double total_seen_sample_num = 0, trained_batches = 0;
  base::Timer t;
  // in batch
  double batch_err = 0; int batch_simple_size = 0;
public:
  TrainStats() : debug(TrainStats::getClassName())
  {
    t.start();
  }

  void updateSample(double err, int sampleSize = 1) {
    batch_err += err;
    batch_simple_size = sampleSize;
    total_seen_sample_num++;
  }

  void newBatch() {
    // init batch vals
    batch_err = 0;
    batch_simple_size = 0;
    trained_batches++;
  }

  string getBatchStats() {
    char s[128];
    sprintf(s, "#%.0lf err:%.2lf e/b:%.2lf",
            trained_batches,
            batch_err,
            batch_err / batch_simple_size);
    return string(s);
  }

  void printTrainEndStats() {
    debug.debug("Training end. Total using %s, iter %.0lf batch, %.0lf samples",
                t.end().c_str(), trained_batches, total_seen_sample_num);
  }

  static string getClassName() {
    return "TrainStats";
  }

};


#endif //PROJECT_TRAINSTATS_H
