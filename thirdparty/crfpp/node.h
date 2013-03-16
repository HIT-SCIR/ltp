//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: node.h 1595 2007-02-24 10:18:32Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_NODE_H__
#define CRFPP_NODE_H__

#include <vector>
#include <cmath>
#include "path.h"
#include "common.h"

#define LOG2               0.69314718055
#define MINUS_LOG_EPSILON  50

namespace CRFPP {
  // log(exp(x) + exp(y));
  //    this can be used recursivly
  // e.g., log(exp(log(exp(x) + exp(y))) + exp(z)) =
  // log(exp (x) + exp(y) + exp(z))
  inline double logsumexp(double x, double y, bool flg) {
    if (flg) return y;  // init mode
    double vmin = _min(x, y);
    double vmax = _max(x, y);
    if (vmax > vmin + MINUS_LOG_EPSILON) {
      return vmax;
    } else {
      return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
  }

  struct Path;

  struct Node {
    unsigned int         x;
    unsigned short int   y;
    double               alpha;
    double               beta;
    double               cost;
    double               bestCost;
    Node                *prev;
    int                 *fvector;
    std::vector<Path *>  lpath;
    std::vector<Path *>  rpath;

    void calcAlpha();
    void calcBeta();
    void calcExpectation(double *expected, double, size_t);

    void clear() {
      x = y = 0;
      alpha = beta = cost = 0.0;
      prev = 0;
      fvector = 0;
      lpath.clear();
      rpath.clear();
    }

    void shrink() {
      std::vector<Path *>(lpath).swap(lpath);
      std::vector<Path *>(rpath).swap(rpath);
    }

    Node(): x(0), y(0), alpha(0.0), beta(0.0),
            cost(0.0), bestCost(0.0), prev(0), fvector(0) {}
  };
}

#endif
