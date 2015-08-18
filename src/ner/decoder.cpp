#include "ner/decoder.h"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"
#include "ner/settings.h"

namespace ltp {
namespace ner {

using strutils::split_by_sep;
using strutils::trim_copy;

NERTransitionConstrain::NERTransitionConstrain(const utility::IndexableSmartMap& alphabet,
    const std::vector<std::string>& includes): T(alphabet.size()) {
  for (size_t i = 0; i < includes.size(); ++ i) {
    const std::string& include = includes[i];
    std::vector<std::string> tokens = split_by_sep(include, "->", 1);
    if (tokens.size() != 2) {
      WARNING_LOG("constrain text \"%s\" is in illegal format.", include.c_str());
      continue;
    }

    int from = alphabet.index(trim_copy(tokens[0]));
    int to = alphabet.index(trim_copy(tokens[1]));
    if (-1 == from || -1 == to) {
      WARNING_LOG("label in constrain text \"%s,%s\" is not in alphabet.",
          trim_copy(tokens[0]).c_str(), trim_copy(tokens[1]).c_str());
    } else {
      rep.insert(from * T + to);
    }
  }
}

bool NERTransitionConstrain::can_tran(const size_t& i, const size_t& j) const {
  if (i >= T || j >= T) {
    return false;
  }

  size_t code = i* T+ j;
  return rep.find(code) != rep.end();
}

size_t NERTransitionConstrain::size(void) const {
  return rep.size();
}

void NERViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
            const framework::ViterbiDecodeConstrain& con,
            std::vector<int>& output) {
  ViterbiDecoder::decode(scm, con, output);
}


void NERViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
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
        if (output[i] == 0 ||
                ((output[i]-1) % __num_pos_types__) == 0 ||
                ((output[i]-1) % __num_pos_types__) == 3) {
          partial_idx.push_back(i);
        }
      }

      calc_partial_probabilities(output, partial_idx, partial_probabilities);

    }
  }
}

}     //  end for namespace ner
}     //  end for namespace ltp

