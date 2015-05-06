#ifndef __LTP_POSTAGGER_POSTAGGER_H__
#define __LTP_POSTAGGER_POSTAGGER_H__

#include "postagger/model.h"
#include "postagger/decode_context.h"
#include "postagger/score_matrix.h"
#include "postagger/instance.h"

namespace ltp {
namespace postagger {

class Postagger {
protected:
  Model * model;

public:
  Postagger();
  ~Postagger();

protected:
  /**
   * extract feature from the instance, store the extracted features in a 
   * Decodecontext class.
   *
   * @param[in] inst   The instance.
   * @param[out] ctx   The decode context result.
   * @param[in] create If create is true, create feature for new feature
   *             in the model otherwise not create.
   */
  void extract_features(const Instance * inst,
      DecodeContext* ctx,
      bool create= false) const;

  /**
   * Cache all the score for the certain instance. The cached results are
   * stored in a Scorematrix.
   *
   * @param[in] inst    the instance
   * @param[in] ctx    the decode context
   * @param[in] use_avg use to specify use average parameter
   * @param[out] scm    the score matrix
   */
  void calculate_scores(const Instance* inst,
      const DecodeContext* ctx,
      bool use_avg,
      ScoreMatrix* scm) const;

  /*
   * build labels string for the instance
   *
   *  @param[in]  inst    the instance
   *  @param[out] tags    the list of tags string
   */
  void build_labels(const Instance* inst, std::vector<std::string>& tags) const;
};

}     //  end for namespace postagger
}     //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_POSTAGGER_H__
