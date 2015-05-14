#include "postagger/decoder.h"
#include "utils/strutils.hpp"
#include "utils/logging.hpp"
#include "utils/sbcdbc.hpp"

namespace ltp {
namespace postagger {

using utility::Bitset;
using utility::IndexableSmartMap;
using strutils::split;
using strutils::chomp;
using strutils::chartypes::sbc2dbc_x;

PostaggerLexiconConstrain::PostaggerLexiconConstrain()
  : words(0), successful(false) {
}

bool PostaggerLexiconConstrain::load(std::istream& is,
    const IndexableSmartMap& labels_alphabet) {

  std::string buffer;
  int num_lines = 1;
  int num_entries = 0;

  while (std::getline(is, buffer)) {
    buffer = chomp(buffer);
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

void PostaggerLexiconConstrain::regist(const std::vector<std::string>* _words) {
  words = _words;
}

bool PostaggerLexiconConstrain::can_emit(const size_t& i, const size_t& j) const {
  if (successful && words) {
    Bitset* entry= rep.get(words->at(i).c_str());
    if (NULL == entry) {
      return true;
    } else {
      return entry->get(j);
    }
  }
  return true;
}

}     //  end for namespace postagger
}     //  end for namespace ltp

