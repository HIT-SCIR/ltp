#ifndef __LTP_SEGMENTOR_INSTANCE_H__
#define __LTP_SEGMENTOR_INSTANCE_H__

#include <iostream>
#include "featurevec.h"
#include "mat.h"
#include "sparsevec.h"

namespace ltp {
namespace segmentor {

class Instance {
public:
  Instance() {}

  ~Instance() {
    cleanup();
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

  int num_predicted_words() {
    return predicted_words.size();
  }

  int num_gold_words() {
    return words.size();
  }

  int num_recalled_words() {
    int len = 0;
    int ret = 0;
    int gold_len = 0, predicted_len = 0;

    for (int i = 0; i < words.size(); ++ i) {
      len += words[i].size();
    }

    for (int i = 0, j = 0; i < words.size() && j < predicted_words.size(); ) {
      if (words[i] == predicted_words[j]) {
        ++ ret;
        gold_len += words[i].size();
        predicted_len += predicted_words[j].size();

        ++ i;
        ++ j;
      } else {
        gold_len += words[i].size();
        predicted_len += predicted_words[j].size();

        ++ i;
        ++ j;

        while (gold_len < len && predicted_len < len) {
          if (gold_len < predicted_len) {
            gold_len += words[i].size();
            ++ i;
          } else if (gold_len > predicted_len) {
            predicted_len += predicted_words[j].size();
            ++ j;
          } else {
            break;
          }
        }
      }
    }

    return ret;
  }

  int cleanup() {
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

    uni_features.dealloc();
    uni_scores.dealloc();
    bi_scores.dealloc();

    features.zero();
    predicted_features.zero();

    return 0;
  }
public:
  std::vector< std::string >  raw_forms;
  std::vector< std::string >  forms;
  std::vector< int >          chartypes;
  std::vector< std::string >  tags;
  std::vector< int >          tagsidx;
  std::vector< std::string >  predicted_tags;
  std::vector< int >          predicted_tagsidx;
  std::vector< std::string >  words;
  std::vector< std::string >  predicted_words;
  std::vector< int >          lexicon_match_state;

  math::SparseVec       features;           /*< the gold features */
  math::SparseVec       predicted_features; /*< the predicted features */

  math::Mat< FeatureVector *> uni_features;
  math::Mat< double > uni_scores;
  math::Mat< double > bi_scores;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGENTOR_INSTANCE_H__
