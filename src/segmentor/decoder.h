#ifndef __LTP_SEGMENTOR_DECODER_H__
#define __LTP_SEGMENTOR_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "utils/math/mat.h"

namespace ltp {
namespace segmentor {

class SegmentationConstrain: public framework::ViterbiDecodeConstrain {
private:
  const std::vector<int>* chartypes;
public:
  SegmentationConstrain();

  void regist(const std::vector<int>* chartypes);

  bool can_tran(const size_t& i, const size_t& j) const;

  bool can_emit(const size_t& i, const size_t& j) const;
};

class PartialSegmentationConstrain: public SegmentationConstrain {
public:
  std::vector<int> mat;
public:
  PartialSegmentationConstrain();

  void append(const int& mask);

  bool can_emit(const size_t& i, const size_t& j) const;
};

class SegmentationViterbiDecoderWithMarginal: public framework::ViterbiDecoderWithMarginal {
public:

  void decode(const framework::ViterbiScoreMatrix& scm,
              const framework::ViterbiDecodeConstrain& con,
              std::vector<int>& output);

  void decode(const framework::ViterbiScoreMatrix& scm,
              const framework::ViterbiDecodeConstrain& con,
              std::vector<int>& output,
              double& sequence_probability,
              std::vector<double>& point_probabilities,
              std::vector<double>& partial_probabilities,
              std::vector<int>& partial_idx,
              bool avg = false,
              size_t last_timestamp = 1);
};

}       //  end for namespace segmentor
}       //  end for namespace ltp
#endif    //  end for __LTP_SEGMENTOR_DECODER_H__
