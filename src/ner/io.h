#ifndef __LTP_NER_IO_H__
#define __LTP_NER_IO_H__

#include <iostream>
#include "framework/io.h"
#include "ner/settings.h"
#include "ner/instance.h"

namespace ltp {
namespace ner {

class NERReader: public framework::LineCountsReader {
private:
  bool with_tag;
  bool trace;
  std::string postag_delimiter;
  std::string netag_delimiter;
public:
  /**
   * Constructor for the NER reader.
   */
  NERReader(std::istream & _ifs,
      bool with_tag= false,
      bool trace= false,
      const std::string& pos_delim = "/",
      const std::string& ne_delim = "#");

  Instance* next();
};

class NERWriter {
public:
  NERWriter(std::ostream & _ofs, bool _sequence_prob = false, bool _marginal_prob = false)
      : ofs(_ofs),
      sequence_prob(_sequence_prob),
      marginal_prob(_marginal_prob) {}

  void write(const Instance* inst);
private:
  std::ostream & ofs;
  bool sequence_prob;
  bool marginal_prob;
};

}       //  end for namespace ner
}       //  end for namespace ltp
#endif    //  end for __LTP_SEGMENTOR_WRITER_H__
