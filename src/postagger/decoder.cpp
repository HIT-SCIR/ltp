#include "postagger/decoder.h"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"
#include "utils/sbcdbc.hpp"

namespace ltp {
namespace postagger {

using utility::Bitset;
using utility::IndexableSmartMap;
using strutils::split;
using strutils::trim;
using strutils::chartypes::sbc2dbc_x;

PostaggerLexiconConstrain::PostaggerLexiconConstrain(
    const std::vector<std::string>& _words,
    const utility::SmartMap<utility::Bitset>& _rep)
  : words(_words), rep(_rep) {
}

bool PostaggerLexiconConstrain::can_emit(const size_t& i, const size_t& j) const {
  Bitset* entry= rep.get(words[i].c_str());
  if (NULL == entry) {
    return true;
  } else {
    return entry->get(j);
  }
}

PostaggerLexicon::PostaggerLexicon(): successful(false) {}

bool PostaggerLexicon::success() const { return successful; }

PostaggerLexiconConstrain PostaggerLexicon::get_con(
    const std::vector<std::string>& words) {
  return PostaggerLexiconConstrain(words, rep);
}

bool PostaggerLexicon::load(std::istream& is,
    const IndexableSmartMap& labels_alphabet) {

  std::string buffer;
  int num_lines = 1;
  int num_entries = 0;

  while (std::getline(is, buffer)) {
    trim(buffer);
    if (buffer.size() == 0) {
      WARNING_LOG("line %4d: empty, can not load constrain",
          num_lines);
      continue;
    }

    Bitset mask;
    std::vector<std::string> tokens = split(buffer);

    int num_tokens = tokens.size();
    if (num_tokens <= 1) {
      WARNING_LOG("line %4d: constrain in illegal format, no postag provided",
          num_lines);
      continue;
    }

    std::string key = strutils::chartypes::sbc2dbc_x(tokens[0]);
    for (int i = 1; i < num_tokens; ++ i) {
      int val = labels_alphabet.index(tokens[i]);

      if (val != -1) {
        bool success = mask.set(val);
        if (false == success) {
          WARNING_LOG("line %4d: failed to compile constrain (%s,%s)",
              num_lines, tokens[i].c_str(), tokens[0].c_str());
        }
      } else {
        WARNING_LOG("line %4d: postag \"%s\" not exist.",
            num_lines, tokens[i].c_str());
      }
    }

    if (!mask.empty()) {
      utility::Bitset* entry = rep.get(key.c_str());

      if (entry) {
        entry->merge(mask);
      } else{
        rep.set(key.c_str(), mask);
      }
      ++ num_entries;
    }

    ++ num_lines;
  }
  return (successful = true);
}

void PostaggerViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm, std::vector<int>& output) {
  ViterbiDecoder::decode(scm, output);
}

void PostaggerViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
            const framework::ViterbiDecodeConstrain& con,
            std::vector<int>& output) {
  ViterbiDecoder::decode(scm, con, output);
}

void PostaggerViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
                                                 std::vector<int>& output,
                                                 double& sequence_probability,
                                                 std::vector<double>& point_probabilities,
                                                 bool avg,
                                                 size_t last_timestamp) {
  ViterbiDecoder::decode(scm, output);

  if (sequence_prob || marginal_prob) {
    init_prob_ctx(scm, avg, last_timestamp);

    if (sequence_prob) {
      calc_sequence_probability(output, sequence_probability);
    }

    if (marginal_prob) {
      calc_point_probabilities(output, point_probabilities);
    }
  }
}

void PostaggerViterbiDecoderWithMarginal::decode(const framework::ViterbiScoreMatrix& scm,
                                                 const framework::ViterbiDecodeConstrain& con,
                                                 std::vector<int>& output,
                                                 double& sequence_probability,
                                                 std::vector<double>& point_probabilities,
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
    }
  }
}

}     //  end for namespace postagger
}     //  end for namespace ltp

