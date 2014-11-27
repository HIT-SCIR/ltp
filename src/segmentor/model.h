#ifndef __LTP_SEGMENTOR_MODEL_H__
#define __LTP_SEGMENTOR_MODEL_H__

#include "framework/serializable.h"
#include "segmentor/featurespace.h"
#include "segmentor/parameter.h"
#include "utils/smartmap.hpp"

namespace ltp {
namespace segmentor {

namespace utils = ltp::utility;
namespace frame = ltp::framework;

class Model: public frame::Serializable {
public:
  Model();
  ~Model();

  /**
   * get number of labels;
   *
   *  @return   int   the number of labels
   */
  inline int num_labels(void)const {
    return labels.size();
  }

  /**
   * save the model to a output stream
   *
   *  @param[out] ofs   the output stream
   */
  void save(std::ostream & ofs);

  /**
   * load the model from an input stream
   *
   *  @param[in]  ifs   the input stream
   */
  bool load(std::istream & ifs);

public:
  //! The timestamp for the last training instance.
  int end_time;

  //! Use to specified if dump the full model.
  bool full;

  //! The feature space.
  FeatureSpace space;

  //! The parameter array.
  Parameters param;

  //! The labels.
  utils::IndexableSmartMap  labels;

  //! The internal lexicon use to extract lexicon features.
  utils::SmartMap<bool> internal_lexicon;

  //! The external lexicon use to extract lexicon features.
  utils::SmartMap<bool> external_lexicon;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp

#endif  //  end for __LTP_SEGMENTOR_MODEL_H__
