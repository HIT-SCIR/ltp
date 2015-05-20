#ifndef __LTP_SEGMENTOR_EXTRACTOR_H__
#define __LTP_SEGMENTOR_EXTRACTOR_H__

#include <iostream>
#include <vector>
#include "segmentor/instance.h"
#include "utils/template.hpp"
#include "utils/strvec.hpp"

namespace ltp {
namespace segmentor {

/**
 * A singleton for extracting features
 *
 */
class Extractor {
public:
  static Extractor& extractor();
  static int num_templates();

  /**
   * Extract first-order features and store the list of string features into
   * a StringVec
   *
   *  @param[in]  inst    The pointer to the instance.
   *  @param[in]  idx     The index of the current form.
   *  @param[out] cache   The cached.
   */
  static int extract1o(const Instance& inst, int idx,
      std::vector< utility::StringVec >& cache);
protected:
  Extractor();
  ~Extractor();
private:
  static std::vector< utility::Template* > templates;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp 

#endif  //  end for __LTP_SEGMENTOR_EXTRACTOR_H__
