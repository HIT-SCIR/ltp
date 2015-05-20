#include "segmentor/decoder.h"
#include "segmentor/preprocessor.h"

namespace ltp {
namespace segmentor {

SegmentorConstrain::SegmentorConstrain(): chartypes(0) {
}

void SegmentorConstrain::regist(const std::vector<int>* _chartypes) {
  chartypes = _chartypes;
}

bool SegmentorConstrain::can_tran(const size_t& i, const size_t& j) const {
  return (((i == 0 || i == 1) && (j == 1 || j == 2))
        || ((i == 2 || i == 3) && (j == 0 || j == 3)));
}

bool SegmentorConstrain::can_emit(const size_t& i, const size_t& j) const {
  if (i == 0 && !(j == 0 || j == 3)) { return false; }

  if (chartypes) {
    int flag = (chartypes->at(i)&0x07);
    if ((flag== Preprocessor::CHAR_ENG) || (flag == Preprocessor::CHAR_URI)) {
      return (j == 3);
    }
  }

  return true;
}

}     //  end for namespace segmentor
}     //  end for namespace ltp

