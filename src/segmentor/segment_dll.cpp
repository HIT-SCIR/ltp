#include "segmentor/segment_dll.h"
#include "segmentor/segmentor.h"
#include "segmentor/settings.h"
//#include "instance.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

class SegmentorWrapper : public ltp::segmentor::Segmentor {
public:
  SegmentorWrapper() :
    beg_tag0(-1),
    beg_tag1(-1) {}

  ~SegmentorWrapper() {}

  bool load(const char * model_file, const char * lexicon_file = NULL) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::segmentor::Model;
    if (!model->load(mfs)) {
      delete model;
      return false;
    }

    if (NULL != lexicon_file) {
      std::ifstream lfs(lexicon_file);

      if (lfs) {
        std::string buffer;
        while (std::getline(lfs, buffer)) {
          buffer = ltp::strutils::chomp(buffer);
          if (buffer.size() == 0) {
            continue;
          }
          model->external_lexicon.set(buffer.c_str(), true);
        }
      }
    }

    // don't need to allocate a decoder
    // one sentence, one decoder
    baseAll = new ltp::segmentor::rulebase::RuleBase(model->labels);

    beg_tag0 = model->labels.index( ltp::segmentor::__b__ );
    beg_tag1 = model->labels.index( ltp::segmentor::__s__ );

    return true;
  }

  int segment(const char * str,
      std::vector<std::string> & words) {
    ltp::segmentor::Instance * inst = new ltp::segmentor::Instance;
    // ltp::strutils::codecs::decode(str, inst->forms);
    int ret = ltp::segmentor::rulebase::preprocess(str,
        inst->raw_forms,
        inst->forms,
        inst->chartypes);

    if (-1 == ret || 0 == ret) {
      delete inst;
      words.clear();
      return 0;
    }

    ltp::segmentor::Segmentor::extract_features(inst);
    ltp::segmentor::Segmentor::calculate_scores(inst, true);

    // allocate a new decoder so that the segmentor support multithreaded
    // decoding. this modification was committed by niuox
    ltp::segmentor::Decoder deco(model->num_labels(), *baseAll);

    deco.decode(inst);
    ltp::segmentor::Segmentor::build_words(inst,
                                           inst->predicted_tagsidx,
                                           words,
                                           beg_tag0,
                                           beg_tag1);

    delete inst;
    return words.size();
  }

  int segment(const std::string & str,
      std::vector<std::string> & words) {
    return segment(str.c_str(), words);
  }
private:
  int beg_tag0;
  int beg_tag1;
};

void * segmentor_create_segmentor(const char * path, const char * lexicon_file) {
  SegmentorWrapper * wrapper = new SegmentorWrapper();

  if (!wrapper->load(path, lexicon_file)) {
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

int segmentor_release_segmentor(void * segmentor) {
  if (!segmentor) {
    return -1;
  }
  delete reinterpret_cast<SegmentorWrapper *>(segmentor);
  return 0;
}

int segmentor_segment(void * segmentor,
    const std::string & str,
    std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }

  SegmentorWrapper * wrapper = 0;
  wrapper = reinterpret_cast<SegmentorWrapper *>(segmentor);
  return wrapper->segment(str.c_str(), words);
}
