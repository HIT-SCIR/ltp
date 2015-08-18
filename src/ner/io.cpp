#include "ner/io.h"
#include "utils/strutils.hpp"
#include "utils/sbcdbc.hpp"
#include "utils/codecs.hpp"
#include "utils/logging.hpp"

namespace ltp {
namespace ner {

using strutils::split;
using strutils::trim;
using strutils::chartypes::sbc2dbc_x;

NERReader::NERReader(std::istream& _is,
    bool _with_tag,
    bool _trace,
    const std::string& _pos_delim,
    const std::string& _ne_delim)
  : postag_delimiter(_pos_delim), netag_delimiter(_ne_delim),
  with_tag(_with_tag), trace(_trace), LineCountsReader(_is) {
}

Instance* NERReader::next() {
  if (is.eof()) {
    return 0;
  }

  cursor ++;
  if (trace && cursor % interval == 0) {
    INFO_LOG("reading: read %d0%% instances.", (cursor/ interval));
  }

  Instance * inst = new Instance;
  std::string line;

  std::getline(is, line);
  trim(line);

  if (line.size() == 0) {
    delete inst;
    return 0;
  }

  std::vector<std::string> words = split(line);
  size_t found;

  for (size_t i = 0; i < words.size(); ++ i) {
    if (with_tag) {
      found = words[i].find_last_of(netag_delimiter);
      if (found == std::string::npos) { delete inst; return 0; }

      std::string tag = words[i].substr(found + 1);
      inst->tags.push_back(tag);
      words[i] = words[i].substr(0, found);

      found = words[i].find_last_of(postag_delimiter);
      if (found == std::string::npos) { delete inst; return 0; }

      std::string postag = words[i].substr(found + 1);
      inst->postags.push_back(postag);

      words[i] = words[i].substr(0, found);
      inst->raw_forms.push_back(words[i]);
      inst->forms.push_back(sbc2dbc_x(words[i]));
    } else {
      found = words[i].find_last_of(postag_delimiter);
      if (found == std::string::npos) { delete inst; return 0; }

      std::string postag = words[i].substr(found + 1);
      inst->postags.push_back(postag);

      words[i] = words[i].substr(0, found);
      inst->raw_forms.push_back(words[i]);
      inst->forms.push_back(sbc2dbc_x(words[i]));
    }
  }
  return inst;
}

void NERWriter::write(const Instance* inst) {
  size_t len = inst->size();
  if (inst->predict_tags.size() != len) {
    return;
  }

  for (size_t i = 0; i < len; ++ i) {
    ofs << inst->forms[i]
      << "/" << inst->postags[i]
      << "#" << inst->predict_tags[i];
    if (i + 1 < len ) {
      ofs << "\t";
    } else {
      ofs << std::endl;
    }
  }

  if (sequence_prob) {
    ofs << inst -> sequence_probability << std::endl;
  }

  if (marginal_prob) {
    for (size_t i = 0; i < len; ++ i) {
      ofs << inst -> point_probabilities[i];
      if (i + 1 < len) {
        ofs << "\t";
      } else {
        ofs << std::endl;
      }
    }

    for (size_t i = 0; i < inst->partial_probabilities.size(); ++ i) {
      if (i + 1 < inst -> partial_probabilities.size()) {
        ofs << "("
            << inst -> partial_idx[i]
            << ","
            << inst -> partial_idx[i+1] - 1
            << "):"
            << inst -> partial_probabilities[i]
            << "\t";
      } else {
        ofs << "("
            << inst -> partial_idx[i]
            << ","
            << inst -> tagsidx.size() - 1
            << "):"
            << inst -> partial_probabilities[i]
            << std::endl;
      }
    }
  }
}


} //  namespace ner
} //  namespace ltp
