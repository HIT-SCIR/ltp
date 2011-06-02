//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: node.cpp 1595 2007-02-24 10:18:32Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <cmath>
#include "node.h"
#include "common.h"

namespace CRFPP {

  void Node::calcAlpha() {
    alpha = 0.0;
    for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it)
      alpha = logsumexp(alpha,
                        (*it)->cost +(*it)->lnode->alpha,
                        (it == lpath.begin()));
    alpha += cost;
  }

  void Node::calcBeta() {
    beta = 0.0;
    for (const_Path_iterator it = rpath.begin(); it != rpath.end(); ++it)
      beta = logsumexp(beta,
                       (*it)->cost +(*it)->rnode->beta,
                       (it == rpath.begin()));
    beta += cost;
  }

  void Node::calcExpectation(double *expected, double Z, size_t size) {
    double c = std::exp(alpha + beta - cost - Z);
    for (int *f = fvector; *f != -1; ++f) expected[*f + y] += c;
    for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it)
      (*it)->calcExpectation(expected, Z, size);
  }
}
