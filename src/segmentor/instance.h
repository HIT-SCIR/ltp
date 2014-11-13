#ifndef __LTP_SEGMENTOR_INSTANCE_H__
#define __LTP_SEGMENTOR_INSTANCE_H__

#include <iostream>
#include "utils/math/mat.h"
#include "utils/math/featurevec.h"
#include "utils/math/sparsevec.h"

namespace ltp {
namespace segmentor {

class Instance {
public:
  Instance() {}

  ~Instance() {
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
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGENTOR_INSTANCE_H__
