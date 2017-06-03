//
// Created by liu on 2017/1/7.
//

#ifndef PROJECT_BICONVERTER_H
#define PROJECT_BICONVERTER_H

#include "Converter.h"
namespace extractor {
  template<class T1, class T2>
  class BiConverter : public Converter<T1, T2> {
  public:
    base::Debug debug;
    BiConverter(): debug("BiConverter") {};

    virtual void iconv(vector<T2> * t2dataPtr = NULL) {
      vector<T2>& d2 = (t2dataPtr == NULL) ? this->data : (*t2dataPtr);
      for (int j = 0; j < d2.size(); ++j) {
        iconvOne((*(this->origin_data_ref))[d2[j].upDataIndex], d2[j], d2[j].upDataInnerIndex );
      }
    };

    virtual void run() {
      unsigned num = this->origin_data_ref->size();
      debug.debug("Convert '%s' to '%s' start. total %u %s items", T1::getClassName().c_str(), T2::getClassName().c_str(), num, T1::getClassName().c_str());
      base::ProgressBar bar(num);
      for (unsigned i = 0; i < num; i++) {
        unsigned startIndex = this->data.size();
        this->convert((*(this->origin_data_ref))[i]);
        unsigned endSize = this->data.size();
        injectUpDataIndex(startIndex, endSize, i);
        if (bar.updateLength(i + 1))
          debug.info("(%d)%s is converted %d data generate.", i + 1, bar.getProgress(i + 1).c_str(), this->data.size());
      }
      debug.debug("Convert '%s' to '%s' finish. generate %u %s items", T1::getClassName().c_str(), T2::getClassName().c_str(), this->data.size(), T2::getClassName().c_str());
    }

    virtual void iconvOne(T1 &t1, T2 &t2, int innerIndex) = 0;

  private:
    void injectUpDataIndex(unsigned startIndex, unsigned endSize, unsigned upIndex) {
      for (int i = startIndex, ii = 0; i < endSize; i++, ii++) {
        this->data[i].upDataIndex = upIndex;
        this->data[i].upDataInnerIndex = ii;
      }
    }
  };
}

#endif //PROJECT_BICONVERTER_H
