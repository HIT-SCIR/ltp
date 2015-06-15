#include "segmentor/partial_segmentation.h"
#include "segmentor/settings.h"
#include "utils/strutils.hpp"

namespace ltp {
namespace segmentor {

using strutils::split;
using strutils::startswith;

int PartialSegmentationUtils::split_by_partial_tag(
    const std::string& line,
    std::vector<std::string>& words) {
  size_t offset1 = line.find(__partial_start__);
  size_t offset2 = line.find(__word_start__);

  if (offset1 == std::string::npos && offset2 == std::string::npos) {
    // 0 representing no partial tags. split with the original tags.
    words = split(line);
    return 0;
  }

  size_t offset = 0;
  size_t prelude = 0, coda = 0, len_start_tag = 0, len_end_tag = 0;

  while (offset < line.length()) {
    prelude = (offset1 < offset2 ? offset1 : offset2);
    std::string word = line.substr(offset, prelude - offset);
    if (word.length() > 0) { words.push_back( word ); }

    if (offset1 < offset2) {
      coda = line.find(__partial_end__, prelude);
      len_start_tag = __partial_start__.length();
      len_end_tag   = __partial_end__.length();
    } else {
      coda = line.find(__word_end__, prelude);
      len_start_tag = __word_start__.length();
      len_end_tag   = __word_end__.length();
    }

    word = line.substr(prelude + len_start_tag, coda - prelude - len_start_tag);

    if ((word.find(__partial_start__) != std::string::npos) ||
        (word.find(__partial_end__)   != std::string::npos) ||
        (word.find(__word_start__)    != std::string::npos) ||
        (word.find(__word_end__)      != std::string::npos)) {
      return -1;
    }

    words.push_back( line.substr(prelude, coda - prelude + len_end_tag) );
    offset = coda + len_end_tag;

    offset1 = line.find(__partial_start__, offset);
    offset2 = line.find(__word_start__, offset);

    if (offset1 == std::string::npos && offset2 == std::string::npos) {
      // 0 representing no partial tags.
      word = line.substr(offset);
      if (word.length() > 0) { words.push_back( word ); }
      break;
    }
  }
  return 1;
}

void PartialSegmentationUtils::trim_partial_tag(const std::string& input, std::string& output) {
  if (startswith(input, __word_start__)) {
    output = input.substr(__word_start__.length(),
        input.length() - __word_start__.length() - __word_end__.length());
  } else if (startswith(input, __partial_start__)) {
    output = input.substr(__partial_start__.length(),
        input.length() - __partial_start__.length() - __partial_end__.length());
  } else {
    output = input;
  }
}

bool PartialSegmentationUtils::is_partial_tagged_word(const std::string& word) {
  return startswith(word, __word_start__);
}


} //  namespace segmentor
} //  namespace ltp
