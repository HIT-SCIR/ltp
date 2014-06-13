#include "ner/ner_dll.h"
#include "ner/ner.h"
#include "ner/settings.h"
#include "utils/logging.hpp"
#include "utils/codecs.hpp"

#include <iostream>

class NERWrapper : public ltp::ner::NER {
public:
  NERWrapper()
    : beg_tag0(-1),
      beg_tag1(-1) {}

  ~NERWrapper() {}

  bool load(const char * model_file) {
    std::ifstream mfs(model_file, std::ifstream::binary);

    if (!mfs) {
      return false;
    }

    model = new ltp::ner::Model;
    if (!model->load(mfs)) {
      delete model;
      return false;
    }

    // beg_tag0 = model->labels.index( );
    // beg_tag1 = model->labels.index( );

    return true;
  }

  int recognize(const std::vector<std::string> & words,
      const std::vector<std::string> & postags,
      std::vector<std::string> & tags) {
    ltp::ner::rulebase::RuleBase base(model->labels);
    ltp::ner::Decoder deco(model->num_labels(), base);

    ltp::ner::Instance * inst = new ltp::ner::Instance;

    for (int i = 0; i < words.size(); ++ i) {
      inst->forms.push_back(ltp::strutils::chartypes::sbc2dbc_x(words[i]));
      inst->postags.push_back(postags[i]);
    }

    ltp::ner::NER::extract_features(inst);
    ltp::ner::NER::calculate_scores(inst, true);
    deco.decode(inst);

    for (int i = 0; i < words.size(); ++ i) {
      tags.push_back(model->labels.at(inst->predicted_tagsidx[i]));
    }

    delete inst;
    return tags.size();
  }

private:
  int beg_tag0;
  int beg_tag1;
};

void * ner_create_recognizer(const char * path) {
  NERWrapper * wrapper = new NERWrapper();

  if (!wrapper->load(path)) {
    return 0;
  }

  return reinterpret_cast<void *>(wrapper);
}

int ner_release_recognizer(void * ner) {
  if (!ner) {
    return -1;
  }
  delete reinterpret_cast<NERWrapper *>(ner);
  return 0;
}

int ner_recognize(void * ner,
    const std::vector<std::string> & words,
    const std::vector<std::string> & postags,
    std::vector<std::string> & tags) {

  if (words.size() != postags.size()) {
    return 0;
  }

  for (int i = 0; i < words.size(); ++ i) {
    if (words[i].empty() || postags.empty()) {
      return 0;
    }
  }

  NERWrapper * wrapper = 0;
  wrapper = reinterpret_cast<NERWrapper *>(ner);
  return wrapper->recognize(words, postags, tags);
}
