#ifndef __LTP_NER_MODEL_H__
#define __LTP_NER_MODEL_H__

#include "featurespace.h"
#include "parameter.h"

#include "smartmap.hpp"

namespace ltp {
namespace ner {

using namespace ltp::utility;

class Model {
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
  IndexableSmartMap   labels;
  FeatureSpace        space;
  Parameters          param;

  SmartMap<int>       cluster_lexicon;
private:
  void write_uint(std::ostream & out, unsigned int val) {
    out.write(reinterpret_cast<const char *>(&val), sizeof(unsigned int));
  }

  unsigned int read_uint(std::istream & in) {
    char p[4];
    in.read(reinterpret_cast<char*>(p), sizeof(unsigned int));
    return *reinterpret_cast<const unsigned int*>(p);
  }
};

}     //  end for namespace ner
}     //  end for namespace ltp

#endif  //  end for __LTP_NER_MODEL_H__
