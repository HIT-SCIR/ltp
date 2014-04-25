#include "postag_dll.h"

#include "postagger.h"
#include "settings.h"

#include "logging.hpp"
#include "codecs.hpp"
#include "sbcdbc.hpp"
#include <iostream>

using namespace ltp::utility;

class PostaggerWrapper : public ltp::postagger::Postagger {
public:
    PostaggerWrapper() {}
    ~PostaggerWrapper() {}

    bool load(const char * model_file, const char * lexicon_file = NULL) {
        std::ifstream mfs(model_file, std::ifstream::binary);

        if (!mfs) {
            return false;
        }

        model = new ltp::postagger::Model;
        if (!model->load(mfs)) {
            delete model;
            return false;
        }

      if (NULL != lexicon_file) {
          std::ifstream lfs(lexicon_file);
          if (lfs) {
              std::string buffer;
              std::vector<std::string> key_values;
              int key_values_size;
              std::string key;
              int value;
              Bitset *  original_bitset;
              while (std::getline(lfs, buffer)) {
                  buffer = ltp::strutils::chomp(buffer);
                  if (buffer.size() == 0) {
                      continue;
                  }
                  Bitset values;
                  key_values = ltp::strutils::split(buffer);
                  key_values_size = key_values.size();
                  key = ltp::strutils::chartypes::sbc2dbc_x(key_values[0]);
                  for(int i=1;i<key_values_size;i++){
                      value = model->labels.index(key_values[i]);
                      if (value != -1){
                          if(!(values.set(value))) {
                              std::cerr << "Tag named " << key_values[i] << " for word "<< key_values[0]<< " add external lexicon error."<<std::endl;
                          }
                      }
                      else {
                          std::cerr << "Tag named " << key_values[i] << " for word "<< key_values[0]<< " is not existed in LTP labels set."<<std::endl;
                      }
                  }
                  if(values.isnotempty()) {
                    original_bitset = model->external_lexicon.get(key.c_str());
                    if(original_bitset){
                      original_bitset->merge(values);
                    }
                    else{
                      model->external_lexicon.set(key.c_str(),values);
                    }
                  }

              }
          }
      }

        return true;
    }

    int postag(const std::vector<std::string> & words,
            std::vector<std::string> & tags) {
        ltp::postagger::Instance * inst = new ltp::postagger::Instance;
        ltp::postagger::Decoder deco(model->num_labels());
        Bitset * original_bitset;

        for (int i = 0; i < words.size(); ++ i) {
            inst->forms.push_back(ltp::strutils::chartypes::sbc2dbc_x(words[i]));
            if( int(model->external_lexicon.size()) != 0){
              original_bitset = model->external_lexicon.get((inst->forms[i]).c_str());
              if(original_bitset){
                inst->external_lexicon_match_state.push_back((*original_bitset));
              }
              else{
                inst->external_lexicon_match_state.push_back(Bitset());
              }
            }
        }

        ltp::postagger::Postagger::extract_features(inst);
        ltp::postagger::Postagger::calculate_scores(inst, true);
        deco.decode(inst);

        ltp::postagger::Postagger::build_labels(inst, tags);

        delete inst;
        return tags.size();
    }
};

void * postagger_create_postagger(const char * path, const char * lexicon_file) {
    PostaggerWrapper * wrapper = new PostaggerWrapper();

    if (!wrapper->load(path, lexicon_file)) {
        return 0;
    }

    return reinterpret_cast<void *>(wrapper);
}

int postagger_release_postagger(void * postagger) {
    if (!postagger) {
        return -1;
    }
    delete reinterpret_cast<PostaggerWrapper *>(postagger);
    return 0;
}

int postagger_postag(void * postagger,
        const std::vector<std::string> & words,
        std::vector<std::string> & tags) {
    // std::cout << "input str = " << str << std::endl;
    PostaggerWrapper * wrapper = 0;
    wrapper = reinterpret_cast<PostaggerWrapper *>(postagger);
    return wrapper->postag(words, tags);
}
