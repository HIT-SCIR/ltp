#include "segmentor/customized_segment_dll.h"
#include "segmentor/customized_segmentor.h"
#include "segmentor/segment_dll.cpp"
#include "segmentor/singleton_model.h"
#include "segmentor/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

namespace seg = ltp::segmentor;

/*class SingletonModel {
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
*/
class CustomizedSegmentorWrapper : public seg::CustomizedSegmentor{
public:
  CustomizedSegmentorWrapper()
    : beg_tag0(-1),
    beg_tag1(-1),
    rule(0) {}

  CustomizedSegmentorWrapper(seg::Model * model)
    : beg_tag0(-1),
    beg_tag1(-1),
    rule(0) {
      baseline_model = model;
    }
  ~CustomizedSegmentorWrapper() {
    if (rule) { delete rule; }
  }

  bool load_customized_segmentor(const char * baseline_model_file, const char * customized_model_file, const char * lexicon_file) {
    std::ifstream bmfs(baseline_model_file, std::ifstream::binary);

    if (!bmfs) {
      return false;
    }

    baseline_model = new seg::Model;
    if (!baseline_model->load(bmfs)) {
      delete baseline_model;
      baseline_model = 0;
      return false;
    }

    if (NULL != customized_model_file) {
       if (!( model = load_model(customized_model_file, lexicon_file))) {
        if (baseline_model) {
          delete baseline_model;
          baseline_model = 0;
        }
        return false;
      }
    }

    beg_tag0 = baseline_model->labels.index(seg::__b__);
    beg_tag1 = baseline_model->labels.index(seg::__s__);
    rule = new seg::rulebase::RuleBase(baseline_model->labels);
    return true;
  }


  int segment(seg::Model * mdl,
      seg::Model* base_mdl,
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
    seg::CustomizedSegmentor::build_lexicon_match_state(mdl, base_mdl,inst);
    seg::Segmentor::extract_features(inst, mdl, ctx);
    seg::Segmentor::extract_features(inst, base_mdl, base_ctx);
    seg::CustomizedSegmentor::calculate_scores(mdl,
        base_mdl,
        inst,
        ctx,
        base_ctx,
        true,
        scm);

    seg::Decoder decoder(mdl->num_labels(), *rule);
    decoder.decode(inst, scm);
    seg::Segmentor::build_words(inst, inst->predicted_tagsidx,
        words, beg_tag0, beg_tag1);

    delete ctx;
    delete base_ctx;
    delete scm;
    delete inst;
    return words.size();
  }

  int segment(const std::string & str,
      std::vector<std::string> & words) {
    return segment(model, baseline_model, str.c_str(), words);
  }

  int segment(const char * model_path,
      const char * lexicon_path,
      const std::string & str,
      std::vector<std::string> & words) {
    seg::Model * mdl = load_model(model_path, lexicon_path);
    if (!mdl) {
      return 0;
    }
    int len = segment(mdl, baseline_model, str.c_str(), words);
    delete mdl;

    return len;
  }

private:
  seg::Model * load_model(const char * model_file,
      const char * lexicon_file = NULL) {

    std::ifstream mfs(model_file, std::ifstream::binary);
    seg::Model * mdl = new seg::Model;
    if (!mdl->load(mfs)) {
      delete mdl;
      return NULL;
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
          mdl->external_lexicon.set(buffer.c_str(), true);
        }
      }
    }

    return mdl;
  }

  void release() {
    if (model) {
      delete model;
    }
  }

private:
  int beg_tag0;
  int beg_tag1;
  seg::rulebase::RuleBase * rule;
};

void * customized_segmentor_create_segmentor() {
  CustomizedSegmentorWrapper * wrapper = new CustomizedSegmentorWrapper(seg::SingletonModel::get_model());
  return reinterpret_cast<void *>(wrapper);
}

void * customized_segmentor_create_segmentor(const char * baseline_model_path, const char * model_path, const char * lexicon_path) {
  CustomizedSegmentorWrapper * wrapper = new CustomizedSegmentorWrapper;

  if (!wrapper->load_customized_segmentor(baseline_model_path, model_path, lexicon_path)) {
    delete wrapper;
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

int customized_segmentor_release_segmentor(void * segmentor) {
  if (!segmentor) {
    return -1;
  }

  delete reinterpret_cast<CustomizedSegmentorWrapper*>(segmentor);
  return 0;
}

int customized_segmentor_segment(void * segmentor,
    const std::string & str, 
    std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }
  CustomizedSegmentorWrapper * wrapper = reinterpret_cast<CustomizedSegmentorWrapper *>(segmentor);
  return wrapper->segment(str, words);
}

int customized_segmentor_segment(void * segmentor,
    const char * model_path,
    const char * lexicon_path,
    const std::string & str,
    std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }

  CustomizedSegmentorWrapper * wrapper = reinterpret_cast<CustomizedSegmentorWrapper *>(segmentor);
  return wrapper->segment(model_path, lexicon_path, str, words);
}


