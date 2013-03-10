//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: feature_cache.cpp 1587 2007-02-12 09:00:36Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <algorithm>
#include "feature_cache.h"

namespace CRFPP {

  void FeatureCache::add(const std::vector<int> &f) {
    int *p = feature_freelist_.alloc(f.size() + 1);
    std::copy(f.begin(), f.end(), p);
    p[f.size()] = -1;   // sentinel
    this->push_back(p);
  }

  void FeatureCache::shrink(std::map<int, int> *old2new) {
    for (size_t i = 0; i < size(); ++i) {
      std::vector<int> newf;
      for (int *f = (*this)[i]; *f != -1; ++f) {
        std::map<int, int>::iterator it = old2new->find(*f);
        if (it != old2new->end()) newf.push_back(it->second);
      }
      newf.push_back(-1);
      std::copy(newf.begin(), newf.end(), (*this)[i]);
    }
    return;
  }
}
