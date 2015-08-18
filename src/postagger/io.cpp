#include "postagger/io.h"
#include "utils/sbcdbc.hpp"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"

namespace ltp {
namespace postagger {

using strutils::chartypes::sbc2dbc_x;
using strutils::rsplit_by_sep;
using strutils::split;
using strutils::trim;
using framework::LineCountsReader;

PostaggerReader::PostaggerReader(std::istream& _is,
    const std::string& _delimiter,
    bool _with_tag,
    bool _trace)
  : delimiter(_delimiter),
  with_tag(_with_tag), trace(_trace), LineCountsReader(_is) {
}

Instance* PostaggerReader::next() {
  if (is.eof()) {
    return 0;
  }

  cursor ++;
  if (trace && cursor % interval == 0) {
    INFO_LOG("reading: read %d0%% instances.", (cursor/ interval));
  }
  Instance* inst = new Instance;
  std::string line;

  std::getline(is, line);
  trim(line);

  if (line.size() == 0) {
    delete inst;
    return 0;
  }

  std::vector<std::string> words = split(line);
  for (size_t i = 0; i < words.size(); ++ i) {
    if (with_tag) {
      std::vector<std::string> sep = rsplit_by_sep(words[i], delimiter, 1);
      if (sep.size() == 2) {
        inst->raw_forms.push_back(sep[0]);
        inst->forms.push_back(sbc2dbc_x(sep[0]));
        inst->tags.push_back(sep[1]);
      } else {
        std::cerr << words[i] << std::endl;
        delete inst;
        return 0;
      }
    } else {
      inst->raw_forms.push_back(words[i]);
      inst->forms.push_back(sbc2dbc_x(words[i]));
    }
  }
  return inst;
}

void PostaggerWriter::write(const Instance* inst) {
  size_t len = inst->size();
  if (inst->predict_tags.size() != len) {
    return;
  }

  for (size_t i = 0; i < len; ++ i) {
    ofs << inst->raw_forms[i] << "/" << inst->predict_tags[i];
    if (i + 1 < len) {
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
  }
}

} //  namespace postagger
} //  namespace ltp
