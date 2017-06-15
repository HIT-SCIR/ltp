//
// Created by liu on 2017/5/11.
//

#ifndef PROJECT_CONDITIONSTOPPER_H
#define PROJECT_CONDITIONSTOPPER_H

#include "../base/debug.h"

namespace process {
  class ConditionStopper {
  private:
    double fast_iter, slow_iter, min_epoth_iter, train_dev_gap, no_train_update_zero, no_save_turn;
    int min_batches_iter;
    double fast_f = 0, slow_f = 0;
    base::Debug debug;
  public:
    ConditionStopper(double fast_iter = 0.3,
                     double slow_iter = 0.1,
                     double min_epoth_iter = 2.0,
                     double train_dev_gap = 0.05,
                     double no_train_update_zero = 0.02,
                     double no_save_turn = 20,
                     int min_batches_iter = 300) :
            fast_iter(fast_iter),
            slow_iter(slow_iter),
            min_epoth_iter(min_epoth_iter),
            train_dev_gap(train_dev_gap),
            no_train_update_zero(no_train_update_zero),
            no_save_turn(no_save_turn),
            min_batches_iter(min_batches_iter),
            debug(ConditionStopper::getClassName())
    { }

    virtual bool auto_end(Performance & trainPerf, Performance & bestDevPerf, int batches, double epoch, int lastSaveTurn) {
      fast_f = fast_iter * trainPerf.fscore() + (1 - fast_iter) * fast_f;
      slow_f = slow_iter * trainPerf.fscore() + (1 - slow_iter) * slow_f;
      if (fast_f != fast_f) fast_f = 0.0; // test nan
      if (slow_f != slow_f) slow_f = 0.0;

      return  batches > min_batches_iter &&
              epoch > min_epoth_iter &&
              lastSaveTurn > no_save_turn &&
              (slow_f > 0.99 || slow_f - bestDevPerf.fscore() > train_dev_gap) &&
              fast_f - slow_f < no_train_update_zero;
    }

    static string getClassName() {
      return "ConditionStopper";
    }

  };
}

#endif //PROJECT_CONDITIONSTOPPER_H
