#ifndef __LTP_SEGMENTOR_SEGMENTOR_H__
#define __LTP_SEGMENTOR_SEGMENTOR_H__

#include "framework/decoder.h"
#include "segmentor/model.h"
#include "segmentor/decoder.h"
#include "segmentor/preprocessor.h"
#include "segmentor/options.h"
#include "segmentor/instance.h"

namespace ltp {
namespace segmentor {

class Segmentor {
protected:
  Model* model;
  Preprocessor preprocessor;
  SegmentationConstrain con;
  static const std::string model_header;
public:
  Segmentor();
  ~Segmentor();

protected:
  /**
   * Extract features from one instance, store the extracted features in a
   * DecodeContext class.
   *
   *  @param[in]  inst    The instance.
   *  @param[out] model   The model.
   *  @param[out] ctx     The decode context result.
   *  @param[in]  create  If create is true, create feature for new feature
   *                      in the model otherwise not create.
   */
  void extract_features(const Instance& inst, Model* mdl,
      framework::ViterbiFeatureContext* ctx,
      bool create = false) const;

  /**
   * Build lexicon match state of the instance
   *
   *  @param[in/out]  inst    The instance.
   */
  virtual void build_lexicon_match_state(
      const std::vector<const Model::lexicon_t*>& lexicons,
      Instance* inst) const;

  /**
   * Cache all the score for the certain instance. The cached results are
   * stored in a ScoreMatrix.
   *
   *  @param[in]  inst      The instance
   *  @param[in]  mdl       The model.
   *  @param[in]  ctx       The decode context.
   *  @param[in]  avg       use to specify use average parameter
   *  @param[out] scm       The score matrix.
   */
  void calculate_scores(const Instance& inst,
      const Model& mdl,
      const framework::ViterbiFeatureContext& ctx,
      bool avg,
      framework::ViterbiScoreMatrix* scm);

  void calculate_scores(const Instance& inst,
      const Model& bs_mdl,
      const Model& mdl,
      const framework::ViterbiFeatureContext& bs_ctx,
      const framework::ViterbiFeatureContext& ctx,
      bool avg,
      framework::ViterbiScoreMatrix* scm);

  /**
   * build words from tags for certain instance
   *
   *  @param[in/out]  inst      the instance
   *  @param[in]      tagsidx   the index of tags
   *  @param[out]     words     the output words
   *  @param[in]      begtag0   first of the word begin tag
   *  @param[in]      begtag1   second of the word begin tag
   */
  void build_words(const std::vector<std::string>& chars,
      const std::vector<int>& tagsidx,
      std::vector<std::string>& words);

  /**
   * Load lexicon from file.
   *
   *  @param[in]  filename    The filename
   *  @param[out] lexicon     The pointer to the lexicon.
   */
  void load_lexicon(const char* filename, Model::lexicon_t* lexicon) const;

  /**
   * Load lexicon from string vector.
   *
   *  @param[in]  filename    The filename
   *  @param[out] lexicon     The pointer to the lexicon.
   */
  void load_lexicon(const std::vector<std::string>& texts,
      Model::lexicon_t* lexicon) const;

};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_SEGMENTOR_H__
