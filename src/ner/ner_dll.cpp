#include "ner/ner_dll.h"
#include "ner/ner.h"
#include "ner/extractor.h"
#include "ner/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"
#include "utils/sbcdbc.hpp"
#include "utils/unordered_set.hpp"
#include "framework/decoder.h"

#include <iostream>
#include <fstream>

class __ltp_dll_ner_wrapper : public ltp::ner::NamedEntityRecognizer {
public:
  __ltp_dll_ner_wrapper() {}
  ~__ltp_dll_ner_wrapper() {}

  bool load(const char* model_file) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::framework::Model(ltp::ner::Extractor::num_templates());
    if (!model->load(ltp::ner::NamedEntityRecognizer::model_header, mfs)) {
      delete model;
      return false;
    }

    std::unordered_set<std::string> ne_types;
    for (size_t i = 0; i < model->num_labels(); ++ i) {
      std::string tag = model->labels.at(i);
      if (tag == ltp::ner::OTHER) { continue; }
      ne_types.insert(tag.substr(1+delimiter.size()));
    }
    build_glob_tran_cons(ne_types);

    return true;
  }

  int recognize(const std::vector<std::string> & words,
      const std::vector<std::string> & postags,
      std::vector<std::string> & tags) {
    tags.clear();

    if (words.size() == 0) {
      return 0;
    }

    if (words.size() != postags.size()) {
      return 0;
    }

    ltp::framework::ViterbiFeatureContext ctx;
    ltp::framework::ViterbiScoreMatrix scm;
    ltp::framework::ViterbiDecoder decoder;
    ltp::ner::Instance inst;

    for (size_t i = 0; i < words.size(); ++ i) {
      inst.forms.push_back(ltp::strutils::chartypes::sbc2dbc_x(words[i]));
      inst.postags.push_back(postags[i]);
    }

    extract_features(inst, &ctx, false);
    calculate_scores(inst, ctx, true, &scm);
    decoder.decode(scm, (*glob_con), inst.predict_tagsidx);

    for (size_t i = 0; i < words.size(); ++ i) {
      tags.push_back(model->labels.at(inst.predict_tagsidx[i]));
    }

    return tags.size();
  }

};

void * ner_create_recognizer(const char * path) {
  __ltp_dll_ner_wrapper* wrapper = new __ltp_dll_ner_wrapper();

  if (!wrapper->load(path)) {
    delete wrapper;
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

int ner_release_recognizer(void * ner) {
  if (!ner) {
    return -1;
  }
  delete reinterpret_cast<__ltp_dll_ner_wrapper *>(ner);
  return 0;
}

int ner_recognize(void * ner,
    const std::vector<std::string> & words,
    const std::vector<std::string> & postags,
    std::vector<std::string> & tags) {

  tags.clear();

  if (words.size() == 0) {
    return 0;
  }

  if (words.size() != postags.size()) {
    return 0;
  }

  for (int i = 0; i < words.size(); ++ i) {
    if (words[i].empty() || postags.empty()) {
      return 0;
    }
  }

  __ltp_dll_ner_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_ner_wrapper*>(ner);
  return wrapper->recognize(words, postags, tags);
}
