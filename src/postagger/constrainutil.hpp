#ifndef __LTP_POSTAGGER_CONSTRAINUTIL_H__
#define __LTP_POSTAGGER_CONSTRAINUTIL_H__

#include <iostream>
#include <fstream>
#include <vector>
#include "postagger/model.h"
#include "postagger/instance.h"
#include "utils/logging.hpp"
#include "utils/tinybitset.hpp"
#include "utils/strutils.hpp"
#include "utils/sbcdbc.hpp"

namespace ltp {
namespace postagger {

/**
 * Function for loading constrain from constrain file.
 *
 *  @param[in]  model             The model
 *  @param[in]  constraints_file  The postag constrain file
 *  @return     int               The number of constraints that are successfully
 *                                loaded. If loading constraints failed, return -1
 */
static int load_constrain(Model * model, const char * constraints_file = NULL) {
  if (NULL == constraints_file) {
    return -1;
  }

  std::ifstream lfs(constraints_file);
  if (!lfs) {
    return -1;
  }

  std::string buffer;
  int num_lines = 1;
  int num_entries = 0;

  while (std::getline(lfs, buffer)) {
    buffer = ltp::strutils::chomp(buffer);
    if (buffer.size() == 0) {
      WARNING_LOG("line %4d: empty, can not load constrain",
          num_lines);
      continue;
    }

    Bitset mask;
    std::vector<std::string> tokens = ltp::strutils::split(buffer);

    int num_tokens = tokens.size();

    if (num_tokens <= 1) {
      WARNING_LOG("line %4d: constrain in illegal format, no postag provided",
          num_lines);
      continue;
    }

    std::string key = strutils::chartypes::sbc2dbc_x(tokens[0]);

    for (int i = 1; i < num_tokens; ++ i) {
      int val = model->labels.index(tokens[i]);

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
      utility::Bitset * entry = model->external_lexicon.get(key.c_str());

      if (entry) {
        entry->merge(mask);
      } else{
        model->external_lexicon.set(key.c_str(), mask);
      }
      ++ num_entries;
    }

    ++ num_lines;
  }

  return num_entries;
}

}
}

#endif  //  end for __LTP_POSTAGGER_CONSTRAINUTIL_H__
