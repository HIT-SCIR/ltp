#include "segmentor/segment_dll.h"
#include "segmentor/segmentor.h"
#include "segmentor/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>
#include <fstream>

class __ltp_dll_segmentor_wrapper: public ltp::segmentor::Segmentor {
private:
  std::vector<const ltp::segmentor::Model::lexicon_t*> lexicons;
public:
  __ltp_dll_segmentor_wrapper() {}
  ~__ltp_dll_segmentor_wrapper() {}

  bool load(const char* model_file, const char * lexicon_file = NULL) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::segmentor::Model;
    if (!model->load(model_header.c_str(), mfs)) {
      delete model;
      model = 0;
      return false;
    }

    if (NULL != lexicon_file) {
      load_lexicon(lexicon_file, &model->external_lexicon);
    }

    lexicons.push_back(&(model->internal_lexicon));
    lexicons.push_back(&(model->external_lexicon));
    return true;
  }

  int segment(const char* str, std::vector<std::string> & words) {
    ltp::framework::ViterbiFeatureContext ctx;
    ltp::framework::ViterbiScoreMatrix scm;
    ltp::framework::ViterbiDecoder decoder;
    ltp::segmentor::Instance inst;
 
    int ret = preprocessor.preprocess(str, inst.raw_forms, inst.forms,
      inst.chartypes);

    if (-1 == ret || 0 == ret) {
      words.clear();
      return 0;
    }

    ltp::segmentor::SegmentationConstrain con;
    con.regist(&(inst.chartypes));
    build_lexicon_match_state(lexicons, &inst);
    extract_features(inst, model, &ctx, false);
    calculate_scores(inst, (*model), ctx, true, &scm);

    // allocate a new decoder so that the segmentor support multithreaded
    // decoding. this modification was committed by niuox
    decoder.decode(scm, con, inst.predict_tagsidx);
    build_words(inst.raw_forms, inst.predict_tagsidx, words);

    return words.size();
  }

  int segment(const std::string& str, std::vector<std::string> & words) {
    return segment(str.c_str(), words);
  }
};

class __ltp_dll_customized_segmentor_wrapper: public ltp::segmentor::Segmentor {
private:
  std::vector<const ltp::segmentor::Model::lexicon_t*> lexicons;
  ltp::segmentor::Model* bs_model;
public:
  __ltp_dll_customized_segmentor_wrapper(): bs_model(0) {}
  ~__ltp_dll_customized_segmentor_wrapper() {
    if (bs_model) { delete bs_model; bs_model = 0; }
  }

  bool load(const char* model1, const char* model2,
      const char * lexicon_file = NULL) {
    std::ifstream mfs(model1, std::ifstream::binary);
    if (!mfs) { return false; }

    model = new ltp::segmentor::Model;
    if (!model->load(model_header.c_str(), mfs)) {
      delete model;
      model = 0;
      return false;
    }

    mfs.close();
    mfs.open(model2);
    if (!mfs) { return false; }

    bs_model = new ltp::segmentor::Model;
    if (!bs_model->load(model_header.c_str(), mfs)) {
      delete model;     model = 0;
      delete bs_model;  bs_model = 0;
      return false;
    }

    if (NULL != lexicon_file) {
      load_lexicon(lexicon_file, &model->external_lexicon);
    }

    lexicons.push_back(&(bs_model->internal_lexicon));
    lexicons.push_back(&(model->internal_lexicon));
    lexicons.push_back(&(model->external_lexicon));
    return true;
  }

  int segment(const char* str, std::vector<std::string> & words) {
    ltp::framework::ViterbiFeatureContext ctx, bs_ctx;
    ltp::framework::ViterbiScoreMatrix scm;
    ltp::framework::ViterbiDecoder decoder;
    ltp::segmentor::Instance inst;
 
    int ret = preprocessor.preprocess(str, inst.raw_forms, inst.forms,
      inst.chartypes);

    if (-1 == ret || 0 == ret) {
      words.clear();
      return 0;
    }

    ltp::segmentor::SegmentationConstrain con;
    con.regist(&(inst.chartypes));
    build_lexicon_match_state(lexicons, &inst);
    extract_features(inst, model, &ctx, false);
    extract_features(inst, bs_model, &bs_ctx, false);
    calculate_scores(inst, (*bs_model), (*model), bs_ctx, ctx, true, &scm);

    // allocate a new decoder so that the segmentor support multithreaded
    // decoding. this modification was committed by niuox
    decoder.decode(scm, con, inst.predict_tagsidx);
    build_words(inst.raw_forms, inst.predict_tagsidx, words);

    return words.size();
  }

  int segment(const std::string& str, std::vector<std::string> & words) {
    return segment(str.c_str(), words);
  }
};

void * segmentor_create_segmentor(const char * path, const char * lexicon_file) {
  __ltp_dll_segmentor_wrapper* wrapper = new __ltp_dll_segmentor_wrapper();

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
  delete reinterpret_cast<__ltp_dll_segmentor_wrapper*>(segmentor);
  return 0;
}

int segmentor_segment(void * segmentor, const std::string & str,
  std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }

  __ltp_dll_segmentor_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_segmentor_wrapper*>(segmentor);
  return wrapper->segment(str.c_str(), words);
}

void * customized_segmentor_create_segmentor(const char * path1,
    const char* path2,
    const char * lexicon_file) {
  __ltp_dll_customized_segmentor_wrapper* wrapper =
    new __ltp_dll_customized_segmentor_wrapper();

  if (!wrapper->load(path1, path2, lexicon_file)) {
    delete wrapper;
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

int customized_segmentor_release_segmentor(void * segmentor) {
  if (!segmentor) {
    return -1;
  }
  delete reinterpret_cast<__ltp_dll_customized_segmentor_wrapper*>(segmentor);
  return 0;
}

int customized_segmentor_segment(void * segmentor, const std::string & str,
  std::vector<std::string> & words) {
  if (str.empty()) {
    return 0;
  }

  __ltp_dll_customized_segmentor_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_customized_segmentor_wrapper*>(segmentor);
  return wrapper->segment(str.c_str(), words);
}
