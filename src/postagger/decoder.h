#ifndef __LTP_POSTAGGER_DECODER_H__
#define __LTP_POSTAGGER_DECODER_H__

#include <iostream>
#include <vector>
#include "framework/decoder.h"
#include "postagger/instance.h"
#include "utils/tinybitset.hpp"
#include "utils/smartmap.hpp"

namespace ltp {
namespace postagger {

class PostaggerLexiconConstrain: public framework::ViterbiDecodeConstrain {
private:
  const utility::SmartMap<utility::Bitset>& rep;
  const std::vector<std::string>& words;
public:
  PostaggerLexiconConstrain(const std::vector<std::string>& words,
      const utility::SmartMap<utility::Bitset>& rep);
  bool can_emit(const size_t& i, const size_t& j) const;
};

class PostaggerLexicon {
private:
  utility::SmartMap<utility::Bitset> rep;
  bool successful;
public:
  PostaggerLexicon();

  bool success() const;
  PostaggerLexiconConstrain get_con(const std::vector<std::string>& words);
  bool load(std::istream& is, const utility::IndexableSmartMap& labels_alphabet);
};


}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_DECODER_H__
