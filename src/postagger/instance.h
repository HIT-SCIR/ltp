#ifndef __LTP_POSTAGGER_INSTANCE_H__
#define __LTP_POSTAGGER_INSTANCE_H__

#include <iostream>
#include "utils/math/mat.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"
#include "utils/tinybitset.hpp"

namespace ltp {
namespace postagger {

using namespace ltp::utility;

class Instance {
public:
  Instance() {}

  ~Instance() {
    if (uni_features.total_size() > 0) {
      int d1 = uni_features.nrows();
      int d2 = uni_features.ncols();

      for (int i = 0; i < d1; ++ i) {
        if (uni_features[i][0]) {
          uni_features[i][0]->clear();
        }
        for (int j = 0; j < d2; ++ j) {
          if (uni_features[i][j]) {
            delete uni_features[i][j];
          }
        }
      }
    }
  }

  inline size_t size() const {
    return forms.size();
  }

  int num_errors() {
    int len = size();
    if ((len != tagsidx.size()) || (len != predicted_tagsidx.size())) {
      return -1;
    }

    int ret = 0;
    for (int i = 0; i < len; ++ i) {
      if (tagsidx[i] != predicted_tagsidx[i]) {
        ++ ret;
      }
    }

    return ret;
  }

  int num_corrected_predicted_tags() {
    return size() - num_errors();
  }

public:
  std::vector< std::string >  raw_forms;
  std::vector< std::string >  forms;
  //std::vector< int >          wordtypes;
  std::vector< std::string >  tags;
  std::vector< int >          tagsidx;
  std::vector< std::string >  predicted_tags;
  std::vector< int >          predicted_tagsidx;

  std::vector< Bitset >       postag_constrain;   /*< the postag constrain for decode */

  math::SparseVec             features;           /*< the gold features */
  math::SparseVec             predicted_features; /*< the predicted features */

  math::Mat< math::FeatureVector *> uni_features;
  math::Mat< double > uni_scores;
  math::Mat< double > bi_scores;

  std::vector< std::vector< std::string> > chars;
};

}     //  end for namespace postagger
}     //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_INSTANCE_H__
