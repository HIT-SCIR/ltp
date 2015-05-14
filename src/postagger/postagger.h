#ifndef __LTP_POSTAGGER_POSTAGGER_H__
#define __LTP_POSTAGGER_POSTAGGER_H__

#include "framework/model.h"
#include "framework/decoder.h"
#include "postagger/instance.h"

namespace ltp {
namespace postagger {

class Postagger {
protected:
  framework::Model* model;
  static const std::string model_header;
public:
  Postagger();
  ~Postagger();

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
      bool create= false) const;

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
   * build labels string for the instance
   *
   *  @param[in]  inst    the instance
   *  @param[out] tags    the list of tags string
   */
  void build_labels(const Instance& inst, std::vector<std::string>& tags) const;
};

}     //  end for namespace postagger
}     //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_POSTAGGER_H__
