#include "postagger/postag_dll.h"
#include "postagger/postagger.h"
#include "postagger/settings.h"
#include "postagger/decoder.h"
#include "postagger/extractor.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"
#include "utils/sbcdbc.hpp"
#include "utils/tinybitset.hpp"
#include <iostream>
#include <fstream>

class __ltp_dll_postagger_wrapper : public ltp::postagger::Postagger {
private:
  ltp::postagger::PostaggerLexicon lex;

public:
  __ltp_dll_postagger_wrapper() {}
  ~__ltp_dll_postagger_wrapper() {}

  bool load(const char* model_file, const char* lexicon_file = NULL) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::framework::Model(ltp::postagger::Extractor::num_templates());
    if (!model->load(ltp::postagger::Postagger::model_header, mfs)) {
      delete model;
      return false;
    }

    if (NULL != lexicon_file) { // MSVC need check this.
      std::ifstream lfs(lexicon_file);
      if (lfs.good()) {
        lex.load(lfs, model->labels);
      }
    }

    return true;
  }

  int postag(const std::vector<std::string> & words,
      std::vector<std::string> & tags) {
    ltp::framework::ViterbiFeatureContext ctx;
    ltp::framework::ViterbiScoreMatrix scm;
    ltp::framework::ViterbiDecoder decoder;
    ltp::postagger::Instance inst;

    inst.forms.resize(words.size());
    for (size_t i = 0; i < words.size(); ++ i) {
      ltp::strutils::chartypes::sbc2dbc_x(words[i], inst.forms[i]);
    }

    extract_features(inst, &ctx, false);
    calculate_scores(inst, ctx, true, &scm);
    if (lex.success()) {
      ltp::postagger::PostaggerLexiconConstrain con = lex.get_con(words);
      decoder.decode(scm, con, inst.predict_tagsidx);
    } else {
      decoder.decode(scm, inst.predict_tagsidx);
    }

    ltp::postagger::Postagger::build_labels(inst, tags);
    return tags.size();
  }
};

void * postagger_create_postagger(const char* path, const char* lexicon_file) {
  __ltp_dll_postagger_wrapper* wrapper = new __ltp_dll_postagger_wrapper();

  if (!wrapper->load(path, lexicon_file)) {
    delete wrapper;
    return 0;
  }
  return reinterpret_cast<void *>(wrapper);
}

int postagger_release_postagger(void * postagger) {
  if (!postagger) {
    return -1;
  }

  delete reinterpret_cast<__ltp_dll_postagger_wrapper*>(postagger);
  return 0;
}

int postagger_postag(void * postagger,
    const std::vector<std::string> & words,
    std::vector<std::string> & tags) {
  if (0 == words.size()) {
    return 0;
  }

  for (int i = 0; i < words.size(); ++ i) {
    if (words[i].empty()) {
      return 0;
    }
  }

  __ltp_dll_postagger_wrapper* wrapper = 0;
  wrapper = reinterpret_cast<__ltp_dll_postagger_wrapper*>(postagger);
  return wrapper->postag(words, tags);
}
