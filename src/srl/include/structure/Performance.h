//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_PERFORMANCE_H
#define PROJECT_PERFORMANCE_H


struct Performance {
  double tp = 0;
  double n_arg = 0;
  double n_parg = 0;

  inline double precision() {
    if (n_parg == 0) return 0.0;
    return tp / n_parg;
  }
  inline double recall() {
    if (n_arg == 0) return 0.0;
    return tp / n_arg;
  }
  inline double fscore() {
    if (n_arg == 0 || n_parg == 0) return 0;
    return 2 * precision() * recall() / (precision() + recall());
  }
  string toString() {
    char buf[64];
    sprintf(buf, " P:%.3lf(%.0lf/%.0lf) R:%.3lf(%.0lf/%.0lf) F:%.3lf",
            precision(), tp , n_parg,
            recall(), tp , n_arg,
            fscore());
    return string(buf);
  }
};

/*
 * perf.precision = perf.tp / perf.n_parg;
 * perf.recall = perf.tp / perf.n_arg;
 * perf.fscore = 2 * perf.precision * perf.recall / (perf.precision + perf.recall);
 */

#endif //PROJECT_PERFORMANCE_H
