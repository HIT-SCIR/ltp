#ifndef __LTP_SEGMENTOR_READER_H__
#define __LTP_SEGMENTOR_READER_H__

#include <iostream>
#include "segmentor/settings.h"
#include "segmentor/instance.h"
#include "segmentor/rulebase.h"
#include "utils/strutils.hpp"
#include "utils/codecs.hpp"

namespace ltp {
namespace segmentor {

class SegmentReader {
public:
  SegmentReader(std::istream & _ifs, bool _segmented = false, int _style = 4) :
    ifs(_ifs),
    segmented(_segmented),
    style(_style) {}

  // read instance from input file line by line
  Instance * next() {
    if (ifs.eof()) {
      return 0;
    }

    Instance * inst = new Instance;
    std::string  line;
    std::getline(ifs, line);
    line = strutils::chomp(line);
    if (line.size() == 0) {
      delete inst;
      return 0;
    }

    if (segmented) {
      std::vector<std::string> words = strutils::split(line);
      inst->words = words;

      for (int i = 0; i < words.size(); ++ i) {
        // std::vector<std::string> chars;
        // int num_chars = codecs::decode(words[i], chars);
        int num_chars = rulebase::preprocess(words[i],
            inst->raw_forms,
            inst->forms,
            inst->chartypes);

        // support different style
        if (style == 2) {
          for (int j = 0; j < num_chars; ++ j) {
            // inst->forms.push_back(chars[j]);
            if (j == 0) {
              inst->tags.push_back( __b__ );
            } else {
              inst->tags.push_back( __i__ );
            }
          }
        } else if (style == 4) {
          for(int j = 0; j < num_chars; ++ j) {
            // inst->forms.push_back(chars[j]);
            if (1 == num_chars) {
              inst->tags.push_back( __s__ );
            } else {
              if (0 == j) {
                inst->tags.push_back( __b__ );
              } else if (num_chars - 1 == j) {
                inst->tags.push_back( __e__ );
              } else {
                inst->tags.push_back( __i__ );
              }
            }
          }
        } else if (style == 6) {
          for (int j = 0; j < num_chars; ++ j) {
            // inst->forms.push_back(chars[j]);

            if (1 == num_chars) {
              inst->tags.push_back( __s__ );
            } else {
              if (0 == j) {
                inst->tags.push_back( __b__ );
              } else if (1 == j) {
                inst->tags.push_back( __b2__ );
              } else if (2 == j) {
                inst->tags.push_back( __b3__ );
              } else if (num_chars - 1 == j) {
                inst->tags.push_back( __e__ );
              } else {
                inst->tags.push_back( __i__ );
              }
            }
          }
        }
      }
    } else {
      int ret = rulebase::preprocess(line,
          inst->raw_forms,
          inst->forms,
          inst->chartypes);

      if (ret < 0) {
        delete inst;
        return 0;
      }
    }

    return inst;
  }
private:
  std::istream& ifs;
  int style;
  bool segmented;
};

}       //  end for namespace segmentor
}       //  end for namespace ltp

#endif    //  end for __LTP_SEGMENTOR_READER_H__
