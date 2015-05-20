#include "ner/decoder.h"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"

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

}     //  end for namespace ner
}     //  end for namespace ltp

