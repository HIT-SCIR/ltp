#include "segmentor/segment_dll.h"
#include "segmentor/segmentor.h"
#include "segmentor/settings.h"
//#include "instance.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

namespace seg = ltp::segmentor;

class SegmentorWrapper : public seg::Segmentor {
public:
  SegmentorWrapper()
    : beg_tag0(-1),
    beg_tag1(-1),
    rule(0) {}

  ~SegmentorWrapper() {
    if (rule) { delete rule; }
  }

  bool load(const char * model_file, const char * lexicon_file = NULL) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new seg::Model;
    if (!model->load(mfs)) {
      delete model;
      model = 0;
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

    beg_tag0 = model->labels.index( seg::__b__ );
    beg_tag1 = model->labels.index( seg::__s__ );

    rule = new seg::rulebase::RuleBase(model->labels);

    return true;
  }

  int segment(const char * str,
      std::vector<std::string> & words) {
    seg::Instance * inst = new seg::Instance;
    // ltp::strutils::codecs::decode(str, inst->forms);
    int ret = seg::rulebase::preprocess(str,
        inst->raw_forms,
        inst->forms,
        inst->chartypes);

    if (-1 == ret || 0 == ret) {
      delete inst;
      words.clear();
      return 0;
    }

    seg::DecodeContext* ctx = new seg::DecodeContext;
    seg::ScoreMatrix* scm = new seg::ScoreMatrix;
    seg::Segmentor::build_lexicon_match_state(inst);
    seg::Segmentor::extract_features(inst, seg::Segmentor::model, ctx);
    seg::Segmentor::calculate_scores(inst, seg::Segmentor::model, ctx, true, scm);

    // allocate a new decoder so that the segmentor support multithreaded
    // decoding. this modification was committed by niuox
    seg::Decoder decoder(model->num_labels(), *rule);
    decoder.decode(inst, scm);
    seg::Segmentor::build_words(inst, inst->predicted_tagsidx,
        words, beg_tag0, beg_tag1);

    delete ctx;
    delete scm;
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

  // don't need to allocate a decoder
  // one sentence, one decoder
  seg::rulebase::RuleBase* rule;

};

void * segmentor_create_segmentor(const char * path, const char * lexicon_file) {
  SegmentorWrapper * wrapper = new SegmentorWrapper();

  if (!wrapper->load(path, lexicon_file)) {
    delete wrapper;
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
