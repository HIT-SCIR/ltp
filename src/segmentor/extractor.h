#ifndef __LTP_SEGMENTOR_EXTRACTOR_H__
#define __LTP_SEGMENTOR_EXTRACTOR_H__

#include <iostream>
#include <vector>
#include "segmentor/instance.h"
#include "utils/template.hpp"
#include "utils/strvec.hpp"

namespace ltp {
namespace segmentor {

using namespace std;
using namespace ltp::utility;

class Extractor {
public:
  static Extractor * extractor();
  static int num_templates();
  static int extract1o(Instance * inst, int idx, vector< StringVec > & cache);
protected:
  Extractor();
  ~Extractor();
private:
  static Extractor * instance_;
  static vector< Template * > templates;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp 

#endif  //  end for __LTP_SEGMENTOR_EXTRACTOR_H__
