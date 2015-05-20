#ifndef __LTP_NER_NER_H__
#define __LTP_NER_NER_H__

#include "framework/decoder.h"
#include "framework/model.h"
#include "ner/instance.h"
#include "ner/decoder.h"
#include "utils/unordered_set.hpp"

namespace ltp {
namespace ner {

class NamedEntityRecognizer {
protected:
  framework::Model* model;  //! The pointer to the model.
  NERTransitionConstrain* glob_con;
  static const std::string model_header;
  static const std::string delimiter;  //! The delimiter between position tag and ne type
public:
  NamedEntityRecognizer();
  ~NamedEntityRecognizer();

protected:
  /**
   * extract feature from the instance, store the extracted features in a
   * framework::ViterbiFeatureContext class.
   *
   *  @param[in]   inst     The instance.
   *  @param[out]  ctx      The decode context result.
   *  @param[in]   create   If create is true, create feature for new feature
   *                        in the model otherwise not create.
   */
  void extract_features(const Instance& inst,
      framework::ViterbiFeatureContext* ctx,
      bool create = false) const;

  /**
   * Cache all the score for the certain instance. The cached results are
   * stored in a Scorematrix.
   *
   *  @param[in]  inst    the instance
   *  @param[in]  ctx     the decode context
   *  @param[in]  avg     use to specify use average parameter
   *  @param[out] scm     the score matrix
   */
  void calculate_scores(const Instance& inst,
      const framework::ViterbiFeatureContext& ctx,
      bool avg,
      framework::ViterbiScoreMatrix* scm) const;

  /**
   * build words from tags for certain instance
   *
   *  @param[in/out]  inst    the instance
   *  @param[out]     words   the output words
   *  @param[in]      tagsidx the index of tags
   *  @param[in]      begtag0 first of the word begin tag
   *  @param[in]      begtag1 second of the word begin tag
   */
  void build_entities(const Instance* inst,
      const std::vector<int>& tagsidx,
      std::vector<std::string>& entities,
      std::vector<std::string> & entities_tags,
      const size_t& delimiter_length = 1) const;

  void build_glob_tran_cons(const std::unordered_set<std::string>& ne_types);
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_NER_NER_H__
