//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_LOOPCOUNTER_H
#define PROJECT_LOOPCOUNTER_H

namespace model {
  class LoopCounter {
    unsigned counter = 0;
  public:
    unsigned size;
    LoopCounter(unsigned size): size(0) {}

    /**
     * +=1
     * @return if overflow
     */
    inline bool update() {
      counter += 1;
      if (counter == size) {
        counter = 0;
        return true;
      }
      return false;
    }

    inline unsigned getCounter() {
      return counter;
    }
  };
}

#endif //PROJECT_LOOPCOUNTER_H
