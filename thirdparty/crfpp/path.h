//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: path.h 1595 2007-02-24 10:18:32Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_PATH_H__
#define CRFPP_PATH_H__

#include <stdlib.h>
#include <vector>
#include "node.h"


namespace CRFPP {
  struct Node;

  struct Path {
    Node   *rnode;
    Node   *lnode;
    int    *fvector;
    double  cost;

    Path(): rnode(0), lnode(0), fvector(0), cost(0.0) {}

    // for CRF
    void calcExpectation(double *expected, double, size_t);
    void add(Node *_lnode, Node *_rnode) ;

    void clear() {
      rnode = lnode = 0;
      fvector = 0;
      cost = 0.0;
    }
  };

  typedef std::vector<Path*>::const_iterator const_Path_iterator;
}
#endif
