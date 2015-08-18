#ifndef __LTP_SEGMENTOR_INSTANCE_H__
#define __LTP_SEGMENTOR_INSTANCE_H__

#include <iostream>
#include "utils/math/mat.h"
#include "utils/math/featurevec.h"
#include "utils/math/sparsevec.h"

namespace ltp {
namespace segmentor {

// input data (x,y) = instance
class Instance {
public:
  Instance() {}

  ~Instance() {
  }

  size_t size() const { return forms.size(); }

public:
  std::vector< std::string >  raw_forms; // raw characters of the input
  std::vector< std::string >  forms; // characters after preprocessing
  std::vector< int >          chartypes; // types of characters, digit, text, punct etc.
  std::vector< int >          lexicon_match_state;
  std::vector< std::string >  tags; // tags of characters, {B I E S}
  std::vector< int >          tagsidx; // int tags
  std::vector< std::string >  predict_tags;
  std::vector< int >          predict_tagsidx;
  std::vector< std::string >  words; // words of the input
  std::vector< std::string >  predict_words;

  double                      sequence_probability;
  std::vector< double >       point_probabilities;
  std::vector< double >       partial_probabilities;
  std::vector< int >          partial_idx;
};

class InstanceUtils {
public:
  /**
   * return the number of tags that predict wrong
   *
   *  @param[in]  answer  The answer word vector.
   *  @return int  the number
   */
  static size_t num_errors(const std::vector<int>& answer,
      const std::vector<int>& predict) {
    if (answer.size() != predict.size()) { return answer.size(); }

    size_t ret = 0;
    for (size_t i = 0; i < answer.size(); ++ i) {
      if (answer[i] != predict[i]) { ++ ret; }
    }

    return ret;
  }

  /**
   * calculate the number of words that predict right
   *  @return int  the number
   */
  static int num_recalled_words(const std::vector<std::string>& answer,
      const std::vector<std::string>& predict) {
    int len = 0, ret = 0;
    int answer_len = 0, predict_len = 0;

    for (size_t i = 0; i < answer.size(); ++ i) { len += answer[i].size(); }

    for (size_t i = 0, j = 0; i < answer.size() && j < predict.size(); ) {
      if (answer[i] == predict[j]) {
        ++ ret;
        answer_len += answer[i].size();
        predict_len += predict[j].size();

        ++ i; ++ j;
      } else {
        answer_len += answer[i].size();
        predict_len += predict[j].size();

        ++ i; ++ j;
        while (answer_len < len && predict_len < len) {
          if (answer_len < predict_len) {
            answer_len += answer[i].size();
            ++ i;
          } else if (answer_len > predict_len) {
            predict_len += predict[j].size();
            ++ j;
          } else {
            break;
          }
        }
      }
    }
    return ret;
  }
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGENTOR_INSTANCE_H__
