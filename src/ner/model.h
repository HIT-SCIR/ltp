#ifndef __LTP_NER_MODEL_H__
#define __LTP_NER_MODEL_H__

#include "framework/serializable.h"
#include "ner/featurespace.h"
#include "ner/parameter.h"
#include "utils/smartmap.hpp"

namespace ltp {
namespace ner {

namespace utils = ltp::utility;
namespace frame = ltp::framework;

class Model : public frame::Serializable {
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
  //!
  utils::IndexableSmartMap labels;

  //!
  FeatureSpace space;

  //!
  Parameters param;

  //!
  utils::SmartMap<int> cluster_lexicon;
};

}     //  end for namespace ner
}     //  end for namespace ltp

#endif  //  end for __LTP_NER_MODEL_H__
