#include "segmentor/decoder.h"
#include "segmentor/preprocessor.h"

namespace ltp {
namespace segmentor {

SegmentationConstrain::SegmentationConstrain(): chartypes(0) {
}

void SegmentationConstrain::regist(const std::vector<int>* _chartypes) {
  chartypes = _chartypes;
}

bool SegmentationConstrain::can_tran(const size_t& i, const size_t& j) const {
  return (((i == __b_id__ || i == __i_id__) && (j == __i_id__ || j == __e_id__))
        || ((i == __e_id__ || i == __s_id__) && (j == __b_id__ || j == __s_id__)));
}

bool SegmentationConstrain::can_emit(const size_t& i, const size_t& j) const {
  if (i == 0 && !(j == __b_id__ || j == __s_id__)) { return false; }

  if (chartypes) {
    int flag = (chartypes->at(i)&0x07);
    if ((flag== Preprocessor::CHAR_ENG) || (flag == Preprocessor::CHAR_URI)) {
      return (j == __s_id__);
    }
  }

  return true;
}

PartialSegmentationConstrain::PartialSegmentationConstrain() {
}

void PartialSegmentationConstrain::append(const int& mask) {
  mat.push_back(mask);
}

bool PartialSegmentationConstrain::can_emit(const size_t& i, const size_t& j) const {
  // A conflicted case is "II<xing>"
  // return (mat[i] & (1 << j)) && SegmentationConstrain::can_emit(i, j);
  return (mat[i] & (1<<j));
}

void SegmentationViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
            const framework::ViterbiDecodeConstrain& con,
            std::vector<int>& output) {
  ViterbiDecoder::decode(scm, con, output);
}


void SegmentationViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
        const framework::ViterbiDecodeConstrain& con,
        std::vector<int>& output,
        double& sequence_probability,
        std::vector<double>& point_probabilities,
        std::vector<double>& partial_probabilities,
        std::vector<int>& partial_idx,
        bool avg,
        size_t last_timestamp) {

  ViterbiDecoder::decode(scm, con, output);

  if (sequence_prob || marginal_prob) {
    init_prob_ctx(scm, con, avg, last_timestamp);

    if (sequence_prob) {
      calc_sequence_probability(output, sequence_probability);
    }

    if (marginal_prob) {
      calc_point_probabilities(output, point_probabilities);

      for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] == __b_id__ || output[i] == __s_id__) {
          partial_idx.push_back(i);
        }
      }

      calc_partial_probabilities(output, partial_idx, partial_probabilities);

    }
  }
}

}     //  end for namespace segmentor
}     //  end for namespace ltp

