#include "segmentor/segment_dll.h"
#include "segmentor/customized_segmentor.h"
#include "segmentor/settings.h"
//#include "instance.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

namespace seg = ltp::segmentor;

class SegmentorWrapper : public seg::CustomizedSegmentor{
public:
  SegmentorWrapper()
    : beg_tag0(-1),
    beg_tag1(-1),
    rule(0) {}

  ~SegmentorWrapper() {
    if (rule) { delete rule; }
  }

  bool load_baseline(const char * model_path, const char * lexicon_path = NULL) {
    if ((seg::CustomizedSegmentor::baseline_model = load_model( model_path, lexicon_path))==NULL) {
      return false;
    }

    return true;
  }

  bool load_customized(const char * model_path, const char * lexicon_path = NULL) {
    if ((seg::CustomizedSegmentor::model = load_model(model_path, lexicon_path))==NULL) {
      return false;
    }

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
    seg::Segmentor::build_lexicon_match_state(seg::CustomizedSegmentor::baseline_model, inst);
    seg::Segmentor::extract_features(inst, seg::CustomizedSegmentor::baseline_model, ctx);
    seg::Segmentor::calculate_scores(inst, seg::CustomizedSegmentor::baseline_model, ctx, true, scm);

    // allocate a new decoder so that the segmentor support multithreaded
    // decoding. this modification was committed by niuox
    seg::Decoder decoder(seg::CustomizedSegmentor::baseline_model->num_labels(), *rule);
    decoder.decode(inst, scm);
    seg::Segmentor::build_words(inst, inst->predicted_tagsidx,
        words, beg_tag0, beg_tag1);

    delete ctx;
    delete scm;
    delete inst;
    return words.size();
  }

  int customized_segment(seg::Model * customized_model,
              const char * str,
              std::vector<std::string> & words) {
    seg::Instance * inst = new seg::Instance;
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
    seg::DecodeContext* base_ctx = new seg::DecodeContext;
    seg::ScoreMatrix* scm = new seg::ScoreMatrix;
    seg::CustomizedSegmentor::build_lexicon_match_state(customized_model, seg::CustomizedSegmentor::baseline_model, inst);
    seg::Segmentor::extract_features(inst, customized_model, ctx);
    seg::Segmentor::extract_features(inst, seg::CustomizedSegmentor::baseline_model, base_ctx);
    seg::CustomizedSegmentor::calculate_scores(customized_model,
        seg::CustomizedSegmentor::baseline_model,
        inst,
        ctx,
        base_ctx,
        true,
        scm);

    seg::Decoder decoder(customized_model->num_labels(), *rule);
    decoder.decode(inst, scm);
    seg::Segmentor::build_words(inst, inst->predicted_tagsidx, words, beg_tag0, beg_tag1);

    delete ctx;
    delete base_ctx;
    delete scm;
    delete inst;
    return words.size();
  }

  int segment(const std::string & str,
      std::vector<std::string> & words) {
    return segment(str.c_str(), words);
  }

  int customized_segment(const std::string & str,
                         std::vector<std::string> &words) {
    return customized_segment(seg::CustomizedSegmentor::model, str.c_str(), words);
  }

  int customized_segment(const char * model_path,
              const char * lexicon_path,
              const std::string & str,
              std::vector<std::string> & words) {
    seg::Model * customized_model = NULL;
    if ((customized_model = load_model(model_path, lexicon_path))==NULL) {
      return 0;
    }
    int len = customized_segment(customized_model, str.c_str(), words);
    delete customized_model;

    return len;
  }
private:
  seg::Model* load_model(const char * model_path, const char * lexicon_path = NULL) {
    if ((NULL == model_path)&&(NULL == lexicon_path)) {
      return NULL;
    }

    seg::Model *mdl = new seg::Model;

    if (NULL != model_path) {
      std::ifstream mfs(model_path, std::ifstream::binary);

      if (mfs) {
        if (!mdl->load(mfs)) {
          delete mdl;
          mdl = 0;
          return NULL;
        }
      } else {
        delete mdl;
        mdl = 0;
        return NULL;
      }
    }

    if (NULL != lexicon_path) {
      std::ifstream lfs(lexicon_path);

      if (lfs) {
        std::string buffer;
        while (std::getline(lfs, buffer)) {
          buffer = ltp::strutils::chomp(buffer);
          if (buffer.size() == 0) {
            continue;
          }
          mdl->external_lexicon.set(buffer.c_str(), true);
        }
      }
    }

    beg_tag0 = mdl->labels.index( seg::__b__ );
    beg_tag1 = mdl->labels.index( seg::__s__ );

    if (!rule) {
      rule = new seg::rulebase::RuleBase(mdl->labels);
    }

    return mdl;
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

  if (!wrapper->load_baseline(path, lexicon_file)) {
    delete wrapper;
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

void * segmentor_create_segmentor(const char * baseline_model_path,
                                  const char * model_path,
                                  const char * lexicon_path) {
  SegmentorWrapper * wrapper = new SegmentorWrapper();

  if (!wrapper->load_baseline(baseline_model_path)) {
    delete wrapper;
    return 0;
  }

  if (!wrapper->load_customized(model_path, lexicon_path)) {
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

int segmentor_customized_segment(void * segmentor,
                                 const std::string & str,
                                 std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }
  SegmentorWrapper * wrapper = reinterpret_cast<SegmentorWrapper*>(segmentor);
  return wrapper->customized_segment(str, words);
}
int segmentor_customized_segment(void * parser,
                      const char * model_path,
                      const char * lexicon_path,
                      const std::string & line,
                      std::vector<std::string> & words) {
  if (line.empty()) {
    return 0;
  }

  SegmentorWrapper * wrapper = 0;
  wrapper = reinterpret_cast<SegmentorWrapper *>(parser);
  return wrapper->customized_segment(model_path, lexicon_path, line.c_str(), words);
}
