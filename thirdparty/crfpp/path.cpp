//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: path.cpp 1587 2007-02-12 09:00:36Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <cmath>
#include "path.h"
#include "common.h"

namespace CRFPP {

  void Path::calcExpectation(double *expected, double Z, size_t size) {
    double c = std::exp(lnode->alpha + cost + rnode->beta - Z);
    for (int *f = fvector; *f != -1; ++f)
      expected[*f + lnode->y * size + rnode->y] += c;
  }

  void Path::add(Node *_lnode, Node *_rnode) {
    lnode = _lnode;
    rnode = _rnode;
    lnode->rpath.push_back(this);
    rnode->lpath.push_back(this);
  }
}
