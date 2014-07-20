#ifndef __LTP_POSTAGGER_IO_H__
#define __LTP_POSTAGGER_IO_H__

#include <iostream>
#include "postagger/settings.h"
#include "postagger/instance.h"
#include "utils/sbcdbc.hpp"
#include "utils/strutils.hpp"

namespace ltp {
namespace postagger {

class PostaggerReader {
public:
  PostaggerReader(istream & _ifs, bool _train = false)
    : ifs(_ifs),
      train(_train) {}

  Instance * next() {
    if (ifs.eof()) {
      return 0;
    }

    Instance * inst = new Instance;
    std::string  line;

    std::getline(ifs, line);
    strutils::chomp(line);

    if (line.size() == 0) {
      delete inst;
      return 0;
    }

    std::vector<std::string> words = split(line);
    for (int i = 0; i < words.size(); ++ i) {
      if (train) {
        std::vector<std::string> sep = strutils::rsplit_by_sep(words[i], "_", 1);
        if (sep.size() == 2) {
          inst->raw_forms.push_back(sep[0]);
          inst->forms.push_back(strutils::chartypes::sbc2dbc_x(sep[0]));
          inst->tags.push_back(sep[1]);
        } else {
          std::cerr << words[i] << std::endl;
          delete inst;
          return 0;
        }
      } else {
        inst->raw_forms.push_back(words[i]);
        inst->forms.push_back(strutils::chartypes::sbc2dbc_x(words[i]));
      }
    }

    return inst;
  }
private:
  istream & ifs;
  int       train;
};

/*
 * postag writer class, use to write postag to a stream.
 */
class PostaggerWriter {
public:
  PostaggerWriter(std::ostream & _ofs) : ofs(_ofs) {}

  void write(const Instance * inst) {
    int len = inst->size();
    if (inst->predicted_tags.size() != len) {
      return;
    }

    for (int i = 0; i < len; ++ i) {
      ofs << inst->raw_forms[i] << "/" << inst->predicted_tags[i];
      if (i + 1 < len ) {
        ofs << "\t";
      } else {
        ofs << std::endl;
      }
    }
  }

  void debug(const Instance * inst, bool show_feat = false) {
  }
private:
  std::ostream & ofs;
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_IO_H__
