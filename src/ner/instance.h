#ifndef __LTP_NER_INSTANCE_H__
#define __LTP_NER_INSTANCE_H__

#include <iostream>
#include "utils/math/mat.h"
#include "utils/math/featurevec.h"
#include "utils/math/sparsevec.h"

namespace ltp {
namespace ner {

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
      return len;
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
    int len = size();
    int ret = 0;

    for (int i = 0; i < len; ++ i) {
      if (tagsidx[i] == predicted_tagsidx[i]) {
        ++ ret;
      }
    }

    return ret;
  }

  int num_gold_entities() {
    int ret = 0;
    if (entities.size() == 0) {
      return size();
    }

    for (int i = 0; i < entities_tags.size(); ++ i) {
      if (entities_tags[i] != "O") {
        ++ ret;
      }
    }

    return ret;
  }

  int num_predicted_entities() {
    int ret = 0;
    if (predicted_entities.size() == 0) {
      return size();
    }

    for (int i = 0; i < predicted_entities_tags.size(); ++ i) {
      if (predicted_entities_tags[i] != "O") {
        ++ ret;
      }
    }

    return ret;
  }

  int num_recalled_entites() {
    int len = 0;
    int ret = 0;
    int gold_len = 0, predicted_len = 0;

    for (int i = 0; i < entities.size(); ++ i) {
      len += entities[i].size();
    }

    for (int i = 0, j = 0; i < entities.size() && j < predicted_entities.size(); ) {
      if ((entities[i] == predicted_entities[j]) && 
          (entities_tags[i] == predicted_entities_tags[j])) {
        if (entities_tags[i] != "O") {
          ++ ret;
        }

        gold_len += entities[i].size();
        predicted_len += predicted_entities[j].size();

        ++ i;
        ++ j;
      } else {
        gold_len += entities[i].size();
        predicted_len += predicted_entities[j].size();

        ++ i;
        ++ j;

        while (gold_len < len && predicted_len < len) {
          if (gold_len < predicted_len) {
            gold_len += entities[i].size();
            ++ i;
          } else if (gold_len > predicted_len) {
            predicted_len += predicted_entities[j].size();
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
  std::vector< std::string >  postags;
  std::vector< std::string >  tags;
  std::vector< int >          tagsidx;
  std::vector< std::string >  predicted_tags;
  std::vector< int >          predicted_tagsidx;
  std::vector< std::string >  entities;
  std::vector< std::string >  entities_tags;
  std::vector< std::string >  predicted_entities;
  std::vector< std::string >  predicted_entities_tags;

  math::SparseVec       features;           /*< the gold features */
  math::SparseVec       predicted_features;     /*< the predicted features */

  math::Mat< math::FeatureVector *> uni_features;
  math::Mat< double >     uni_scores;
  math::Mat< double >     bi_scores;
};

}     //  end for namespace ner
}     //  end for namespace ltp

#endif  //  end for __LTP_NER_INSTANCE_H__
