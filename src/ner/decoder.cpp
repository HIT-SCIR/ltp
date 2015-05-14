#include "ner/decoder.h"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"

namespace ltp {
namespace ner {

using strutils::split_by_sep;
using strutils::chomp;

NERTransitionConstrain::NERTransitionConstrain(const utility::IndexableSmartMap& alphabet,
    const std::vector<std::string>& includes): T(alphabet.size()) {
  for (size_t i = 0; i < includes.size(); ++ i) {
    const std::string& include = includes[i];
    std::vector<std::string> tokens = split_by_sep(include, "->", 1);
    if (tokens.size() != 2) {
      WARNING_LOG("constrain text \"%s\" is in illegal format.", include.c_str());
      continue;
    }

    int from = alphabet.index(chomp(tokens[0]));
    int to = alphabet.index(chomp(tokens[1]));
    if (-1 == from || -1 == to) {
      WARNING_LOG("label in constrain text \"%s\" is not in alphabet.", include.c_str());
    }

    rep.insert(from * T + to);
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

