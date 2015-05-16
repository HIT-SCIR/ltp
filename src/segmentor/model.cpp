#include "segmentor/model.h"
#include "segmentor/extractor.h"
#include <cstring>

namespace ltp {
namespace segmentor {

using framework::Parameters;

Model::Model(): framework::Model(Extractor::num_templates()){}
Model::~Model() {}

void Model::save(const char* model_name, const Parameters::DumpOption& opt,
    std::ostream& ofs) {
  framework::Model::save(model_name, opt, ofs);
  internal_lexicon.dump(ofs);
}

bool Model::load(const char* model_name, std::istream& ifs) {
  if (!framework::Model::load(model_name, ifs)) {
    return false;
  }
  if (!internal_lexicon.load(ifs)) {
     return false;
  }

  return true;
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
