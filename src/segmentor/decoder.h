#ifndef __LTP_SEGMENTOR_DECODER_H__
#define __LTP_SEGMENTOR_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "utils/math/mat.h"

namespace ltp {
namespace segmentor {

class SegmentorConstrain: public framework::ViterbiDecodeConstrain {
private:
  const std::vector<int>* chartypes;
public:
  SegmentorConstrain();

  void regist(const std::vector<int>* chartypes);

  bool can_tran(const size_t& i, const size_t& j) const;

  bool can_emit(const size_t& i, const size_t& j) const;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp
#endif    //  end for __LTP_SEGMENTOR_DECODER_H__
