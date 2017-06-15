//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_RANDOMORDERMAP_H
#define PROJECT_RANDOMORDERMAP_H

#include "algorithm"
#include "../base/debug.h"
#include "vector"
using namespace std;
namespace model {
  class RandomOrderMap {
    vector<unsigned> order;
    unsigned size;
    base::Debug debug;
  public:
    RandomOrderMap(unsigned size): order(size), size(size), debug("RandomOrderMap"){
      for (int j = 0; j < size; ++j) {
        order[j] = j;
      }
      shuffle();
    }

    inline unsigned operator[](unsigned index) {
      return order[index];
    }

    void shuffle() {
      debug.debug("**SHUFFLE");
      random_shuffle(order.begin(), order.end());
    }

    inline unsigned operator++(int) {
      static unsigned innerCounter = 0;
      if (innerCounter >= size) {
        innerCounter = 0;
        shuffle();
      }
      return operator[](innerCounter++);
    }

  };
}

#endif //PROJECT_RANDOMORDERMAP_H
