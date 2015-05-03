#ifndef __LTP_POSTAGGER_MODEL_H__
#define __LTP_POSTAGGER_MODEL_H__

#include "framework/serializable.h"
#include "postagger/featurespace.h"
#include "postagger/parameter.h"
#include "utils/smartmap.hpp"
#include "utils/tinybitset.hpp"

namespace ltp {
namespace postagger {

namespace utils = ltp::utility;

class Model: public framework::Serializable {
public:
  Model();
  ~Model();

  /*
   * get number of labels;
   *
   *  @return   int   the number of labels
   */
  inline int num_labels(void) {
    return labels.size();
  }

  /*
   * save the model to a output stream
   *
   *  @param[out] ofs   the output stream
   */
  void save(std::ostream & ofs);

  /*
   * load the model from an input stream
   *
   *  @param[in]  ifs   the input stream
   */
  bool load(std::istream & ifs);
public:
  //! The labels.
  utils::IndexableSmartMap labels;

  //! The feature space.
  FeatureSpace space;

  //! The parameters.
  Parameters param;

  //SmartMap<Bitset>  internal_lexicon;
  utils::SmartMap<utils::Bitset> external_lexicon;
};

}     //  end for namespace postagger
}     //  end for namespace ltp

#endif  //  end for __LTP_POSTAGGER_MODEL_H__
