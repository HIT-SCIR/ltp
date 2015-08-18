#ifndef __LTP_SEGMENTOR_IO_CPP__
#define __LTP_SEGMENTOR_IO_CPP__

#include "segmentor/io.h"
#include "utils/strutils.hpp"
#include "utils/codecs.hpp"
#include "utils/logging.hpp"

namespace ltp {
namespace segmentor {

using framework::LineCountsReader;
using strutils::trim;

SegmentReader::SegmentReader(std::istream& _ifs,
    const Preprocessor& processor,
    bool _segmented, bool _trace)
  : segmented(_segmented), trace(_trace), preprocessor(processor), 
  LineCountsReader(_ifs) {
}

Instance* SegmentReader::next() {
  if (is.eof()) {
    return 0;
  }

  cursor ++;
  if (trace && cursor % interval == 0) {
    INFO_LOG("reading: read %d0%% instances.", (cursor/ interval));
  }

  Instance* inst = new Instance;
  std::string line;

  std::getline(is, line);
  trim(line);

  if (line.size() == 0) {
    delete inst;
    return 0;
  }

  if (segmented) {
    std::vector<std::string> words = strutils::split(line);
    inst->words = words;

    for (size_t i = 0; i < words.size(); ++ i) {
      // std::vector<std::string> chars;
      // int num_chars = codecs::decode(words[i], chars);
      int num_chars = preprocessor.preprocess(words[i], inst->raw_forms,
          inst->forms, inst->chartypes);

      if (num_chars < 0) {
        delete inst;
        return 0;
      }

      for (size_t j = 0; j < num_chars; ++ j) {
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
    }
  } else {
    int ret = preprocessor.preprocess(line, inst->raw_forms,
        inst->forms, inst->chartypes);

    if (ret < 0) {
      delete inst;
      return 0;
    }
  }
  return inst;
}


void SegmentWriter::write(const Instance* inst) {
  size_t len = inst->predict_words.size();
  for (size_t i = 0; i < len; ++ i) {
    ofs << inst->predict_words[i];
    if (i+1==len) ofs << std::endl;
    else ofs << "\t";
  }

  if (sequence_prob) {
    ofs << inst -> sequence_probability << std::endl;
  }

  if (marginal_prob) {
    for (size_t i = 0; i < len; ++ i) {
      ofs << inst -> point_probabilities[i];
      if (i + 1 < len) {
        ofs << "\t";
      } else {
        ofs << std::endl;
      }
    }

    for (size_t i = 0; i < inst->partial_probabilities.size(); ++ i) {
      if (i + 1 < inst -> partial_probabilities.size()) {
        ofs << "("
            << inst -> partial_idx[i]
            << ","
            << inst -> partial_idx[i+1] - 1
            << "):"
            << inst -> partial_probabilities[i]
            << "\t";
      } else {
        ofs << "("
            << inst -> partial_idx[i]
            << ","
            << inst -> tagsidx.size() - 1
            << "):"
            << inst -> partial_probabilities[i]
            << std::endl;
      }
    }
  }
}

void SegmentWriter::debug(const Instance* inst) {
  size_t len = inst->size();
  ofs << "_instance_debug_" << std::endl;
  ofs << "FORMS: ";
  const std::vector<std::string>& forms = inst->forms;
  for (size_t i = 0; i < len; ++ i) { ofs << forms[i] << "|"; } ofs << std::endl;

  ofs << "TAGS: ";
  const std::vector<std::string>& tags = inst->tags;
  for (size_t i = 0; i < tags.size(); ++ i) { ofs << tags[i] << "|"; } ofs << std::endl;

  ofs << "TAGS(index): ";
  const std::vector<int>& tagsidx = inst->tagsidx;
  for (size_t i = 0; i < tagsidx.size(); ++ i) { ofs << tagsidx[i] << "|"; } ofs << std::endl;

  ofs << "predict TAGS: ";
  const std::vector<std::string>& predict_tags = inst->predict_tags;
  for (size_t i = 0; i < predict_tags.size(); ++ i) { ofs << predict_tags[i] << "|"; }
  ofs << std::endl;

  ofs << "predict TAGS(index): ";
  const std::vector<int>& predict_tagsidx = inst->predict_tagsidx;
  for (size_t i = 0; i < predict_tagsidx.size(); ++ i) { ofs << predict_tagsidx[i] << "|"; }
  ofs << std::endl;

  ofs << "WORDS: ";
  const std::vector<std::string>& words = inst->words;
  for (size_t i = 0; i < words.size(); ++ i) { ofs << words[i] << "|"; } ofs << std::endl;

  ofs << "predict WORDS: ";
  const std::vector<std::string>& predict_words = inst->predict_words;
  for (size_t i = 0; i < predict_words.size(); ++ i) { ofs << predict_words[i] << "|"; }
  ofs << std::endl;
}

} //  namespace segmentor
} //  namespace ltp

#endif  //  end for __LTP_SEGMENTOR_IO_CPP__
