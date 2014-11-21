#include "segmentor/customized_segment_dll.h"
#include "segmentor/customized_segmentor.h"
#include "segmentor/segment_dll.cpp"
#include "segmentor/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

namespace seg = ltp::segmentor;

class SingletonModel {
public:
  static seg::Model * get_model(const char * model_file) {
    if (!model) {
      std::ifstream mfs(model_file, std::ifstream::binary);
      if (!mfs) {
        return NULL;
      }
      model = new seg::Model;
      if (!model->load(mfs)) {
        delete model;
        model = 0;
        return NULL;
      }
    }

    return model;
  }
private:
  SingletonModel() {
  }
private:
  static seg::Model * model;
};

seg::Model* SingletonModel::model = NULL;

class CustomizedSegmentorWrapper : public seg::CustomizedSegmentor{
public:

  bool load(const char * baseline_model_file, const char * customized_model_file, const char * lexicon_file) {
    seg::Model * temp = NULL;
    if ((temp=SingletonModel::get_model(baseline_model_file))) {
      baseline_model = temp;
    } else {
      return false;
    }
    std::ifstream mfs(customized_model_file, std::ifstream::binary);

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

      if(lfs) {
        std::string buffer;
        while(std::getline(lfs, buffer)) {
          buffer = ltp::strutils::chomp(buffer);
          if (buffer.size() == 0) {
            continue;
          }
          model->external_lexicon.set(buffer.c_str(), true);
        }
      }
    }
  }

  int segment(const char * str,
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

    seg::rulebase::RuleBase* rule = new seg::rulebase::RuleBase(model->labels);

    int beg_tag0 = model->labels.index( seg::__b__);
    int beg_tag1 = model->labels.index( seg::__s__);
    seg::DecodeContext* ctx = new seg::DecodeContext;
    seg::DecodeContext* base_ctx = new seg::DecodeContext;
    seg::ScoreMatrix* scm = new seg::ScoreMatrix;
    seg::CustomizedSegmentor::build_lexicon_match_state(inst);
    seg::Segmentor::extract_features(inst, seg::Segmentor::model, ctx);
    seg::Segmentor::extract_features(inst, seg::CustomizedSegmentor::baseline_model, base_ctx);
    seg::CustomizedSegmentor::calculate_scores(inst, ctx, base_ctx, true, scm);
    seg::Decoder decoder(model->num_labels(), *rule);
    decoder.decode(inst, scm);
    seg::Segmentor::build_words(inst, inst->predicted_tagsidx,
        words, beg_tag0, beg_tag1);

    delete ctx;
    delete base_ctx;
    delete scm;
    delete inst;
    delete rule;
    return words.size();
  }

  int segment(const std::string & str,
      std::vector<std::string> & words) {
    return segment(str.c_str(), words);
  }

  void release() {
    if (model) {
      delete model;
    }
  }

};

int customized_segmentor_segment(const std::string & baseline_model_path,
    const std::string & model_path,
    const std::string & lexicon_file,
    const std::string & str,
    std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }

  CustomizedSegmentorWrapper * wrapper = new CustomizedSegmentorWrapper;
  if (!wrapper->load(baseline_model_path.c_str(), model_path.c_str(), lexicon_file.c_str())) {
    wrapper->release();
  }
  int len = wrapper->segment(str, words);
  wrapper->release();

  return len;
}


