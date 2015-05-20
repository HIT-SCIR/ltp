#ifndef __LTP_NER_EXTRACTOR_H__
#define __LTP_NER_EXTRACTOR_H__

#include <iostream>
#include <vector>
#include "ner/instance.h"
#include "utils/template.hpp"
#include "utils/strvec.hpp"

namespace ltp {
namespace ner {

class Extractor {
public:
  static Extractor& extractor();
  static int num_templates();
  static int extract1o(const Instance& inst, int idx,
      std::vector<utility::StringVec> & cache);

protected:
  Extractor();
  ~Extractor();
private:
  static std::vector<utility::Template *> templates;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp 

#endif  //  end for __LTP_NER_EXTRACTOR_H__
