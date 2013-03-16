//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: feature_cache.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_FEATURE_CACHE_H__
#define CRFPP_FEATURE_CACHE_H__

#include <vector>
#include <map>
#include "freelist.h"

namespace CRFPP {

  class FeatureCache: public std::vector <int *> {
  private:
    FreeList<int> feature_freelist_;

  public:
    void clear() {
      std::vector<int *>::clear();
      feature_freelist_.free();
    }

    void add(const std::vector<int> &);
    void shrink(std::map<int, int> *);

    explicit FeatureCache(): feature_freelist_(8192 * 16) {}
    virtual ~FeatureCache() {}
  };
}
#endif
