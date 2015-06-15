#ifndef __LTP_PARTIAL_SEGMENTATION_H__
#define __LTP_PARTIAL_SEGMENTATION_H__

#include <iostream>
#include <vector>

namespace ltp {
namespace segmentor {

class PartialSegmentationUtils {
public:
  static int split_by_partial_tag(const std::string& line,
      std::vector<std::string>& words);

  static bool is_partial_tagged_word(const std::string& word);

  static void trim_partial_tag(const std::string& input,
      std::string& output);
};

} //  end namespace segmentor
} //  end namespace ltp

#endif  //  end for __LTP_PARTIAL_SEGMENTATION_H__
