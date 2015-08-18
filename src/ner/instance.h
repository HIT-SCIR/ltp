#ifndef __LTP_NER_INSTANCE_H__
#define __LTP_NER_INSTANCE_H__

#include <iostream>
#include "ner/settings.h"
#include "utils/math/mat.h"
#include "utils/math/featurevec.h"
#include "utils/math/sparsevec.h"

namespace ltp {
namespace ner {

class Instance {
public:
  Instance() {}
  ~Instance() {}

  inline size_t size() const {
    return forms.size();
  }

  size_t num_errors() const {
    size_t len = size();
    if ((len != tagsidx.size()) || (len != predict_tagsidx.size())) {
      return len;
    }

    size_t ret = 0;
    for (size_t i = 0; i < len; ++ i) {
      if (tagsidx[i] != predict_tagsidx[i]) { ++ ret; }
    }

    return ret;
  }

  size_t num_corrected_predict_tags() const {
    size_t len = size();
    size_t ret = 0;

    for (size_t i = 0; i < len; ++ i) {
      if (tagsidx[i] == predict_tagsidx[i]) { ++ ret; }
    }

    return ret;
  }

  size_t num_gold_entities() const {
    size_t ret = 0;
    if (entities.size() == 0) { return size(); }

    for (size_t i = 0; i < entities_tags.size(); ++ i) {
      if (entities_tags[i] != OTHER) { ++ ret; }
    }

    return ret;
  }

  size_t num_predict_entities() const {
    size_t ret = 0;
    if (predict_entities.size() == 0) { return size(); }

    for (size_t i = 0; i < predict_entities_tags.size(); ++ i) {
      if (predict_entities_tags[i] != OTHER) { ++ ret; }
    }

    return ret;
  }

  size_t num_recalled_entities() const {
    size_t len = 0;
    size_t ret = 0;
    size_t gold_len = 0, predict_len = 0;

    for (size_t i = 0; i < entities.size(); ++ i) { len += entities[i].size(); }

    for (size_t i = 0, j = 0; i < entities.size() && j < predict_entities.size(); ) {
      if ((entities[i] == predict_entities[j]) &&
          (entities_tags[i] == predict_entities_tags[j])) {
        if (entities_tags[i] != OTHER) {
          ++ ret;
        }

        gold_len += entities[i].size();
        predict_len += predict_entities[j].size();

        ++ i;
        ++ j;
      } else {
        gold_len += entities[i].size();
        predict_len += predict_entities[j].size();

        ++ i;
        ++ j;

        while (gold_len < len && predict_len < len) {
          if (gold_len < predict_len) {
            gold_len += entities[i].size();
            ++ i;
          } else if (gold_len > predict_len) {
            predict_len += predict_entities[j].size();
            ++ j;
          } else {
            break;
          }
        }
      }
    }

    return ret;
  }

public:
  std::vector< std::string >  raw_forms;
  std::vector< std::string >  forms;
  std::vector< std::string >  postags;
  std::vector< std::string >  tags;
  std::vector< int >          tagsidx;
  std::vector< std::string >  predict_tags;
  std::vector< int >          predict_tagsidx;
  std::vector< std::string >  entities;
  std::vector< std::string >  entities_tags;
  std::vector< std::string >  predict_entities;
  std::vector< std::string >  predict_entities_tags;

  double                      sequence_probability;
  std::vector< double >       point_probabilities;
  std::vector< double >       partial_probabilities;
  std::vector< int >          partial_idx;
};

}     //  end for namespace ner
}     //  end for namespace ltp

#endif  //  end for __LTP_NER_INSTANCE_H__
