#ifndef __LTP_POSTAGGER_IO_H__
#define __LTP_POSTAGGER_IO_H__

#include <iostream>
#include "framework/io.h"
#include "postagger/settings.h"
#include "postagger/instance.h"

namespace ltp {
namespace postagger {

class PostaggerReader: public framework::LineCountsReader {
private:
  std::string delimiter;
  bool with_tag;
  bool trace;
public:
  PostaggerReader(std::istream& _ifs, const std::string& delimiter = "_",
      bool with_tag = false, bool trace = false);

  Instance * next();
};

class PostaggerWriter {
private:
  std::ostream& ofs;
  bool sequence_prob;
  bool marginal_prob;
public:
  PostaggerWriter(std::ostream & _ofs, bool _sequence_prob = false, bool _marginal_prob = false)
      :ofs(_ofs), sequence_prob(_sequence_prob), marginal_prob(_marginal_prob) {}
  void write(const Instance* inst);
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif    //  end for __LTP_POSTAGGER_IO_H__
