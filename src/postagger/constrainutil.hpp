#ifndef __LTP_POSTAGGER_CONSTRAINUTIL_H__
#define __LTP_POSTAGGER_CONSTRAINUTIL_H__

#include <iostream>
#include <fstream>
#include <vector>
#include "model.h"
#include "instance.h"
#include "logging.hpp"
#include "tinybitset.hpp"
#include "strutils.hpp"
#include "sbcdbc.hpp"

namespace ltp {
namespace postagger {
class Constrain{
public:
  static void load_model_constrain(Model * model , const char * lexicon_file = NULL) {
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
                if(key_values_size == 0 || key_values_size == 1) {
                  continue;
                }
                key = ltp::strutils::chartypes::sbc2dbc_x(key_values[0]);
                for(int i=1;i<key_values_size;i++){
                    value = model->labels.index(key_values[i]);
                    if (value != -1){
                        if(!(values.set(value))) {
                            WARNING_LOG("Tag named %s for word %s add external lexicon error.",key_values[i].c_str(),key_values[0].c_str());
                        }
                    }
                    else {
                        WARNING_LOG("Tag named %s for word %s is not existed in LTP labels set.",key_values[i].c_str(),key_values[0].c_str());
                    }
                }
                if(!values.empty()) {
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
  }//end func load_model_constrain
  static void load_inst_constrain(Instance * inst,Bitset *  original_bitset) {
      if(original_bitset){
        inst->external_lexicon_match_state.push_back((*original_bitset));
      }
      else{
        Bitset bitset;
        bitset.allsetones();
        inst->external_lexicon_match_state.push_back(bitset);
      }
  }//end func load_inst_constrain
};
}
}

#endif