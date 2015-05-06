#ifndef __LTP_POSTAGGER_IO_H__
#define __LTP_POSTAGGER_IO_H__

#include <iostream>
#include "framework/io.h"
#include "postagger/settings.h"
#include "postagger/instance.h"

namespace ltp {
namespace postagger {

class PostaggerReader: public framework::Reader {
private:
  std::string delimiter;
  bool with_tag;
  bool trace;
  size_t nr_lines;
  size_t cursor;
  size_t interval;
public:
  PostaggerReader(std::istream& _ifs, const std::string& delimiter = "_",
      bool with_tag = false, bool trace = false);

  Instance * next();
};

class PostaggerWriter {
private:
  std::ostream& ofs;
public:
  PostaggerWriter(std::ostream & _ofs);
  void write(const Instance* inst);
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_IO_H__
