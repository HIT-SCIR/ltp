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
  // write a signature into the file
  char chunk[128];
  ofs.write(chunk, 128);

  int off = ofs.tellp();
  unsigned labels_offset    = 0;
  unsigned lexicon_offset   = 0;
  unsigned feature_offset   = 0;
  unsigned parameter_offset   = 0;

  write_uint(ofs, 0); //  the label offset
  write_uint(ofs, 0); //  the internal lexicon offset
  write_uint(ofs, 0); //  the features offset
  write_uint(ofs, 0); //  the parameter offset

  labels_offset = ofs.tellp();
  labels.dump(ofs);

  lexicon_offset = ofs.tellp();
  internal_lexicon.dump(ofs);

  feature_offset = ofs.tellp();
  space.dump(ofs);

  parameter_offset = ofs.tellp();
  param.dump(ofs, opt);

  ofs.seekp(off);
  write_uint(ofs, labels_offset);
  write_uint(ofs, lexicon_offset);
  write_uint(ofs, feature_offset);
  write_uint(ofs, parameter_offset);
}

bool Model::load(const char* model_name, std::istream& ifs) {
  char chunk[128];
  ifs.read(chunk, 128);

  if (strcmp(chunk, model_name)) {
    return false;
  }

  unsigned labels_offset    = read_uint(ifs);
  unsigned lexicon_offset   = read_uint(ifs);
  unsigned feature_offset   = read_uint(ifs);
  unsigned parameter_offset = read_uint(ifs);

  ifs.seekg(labels_offset);
  if (!labels.load(ifs)) {
    return false;
  }

  ifs.seekg(lexicon_offset);
  if (!internal_lexicon.load(ifs)) {
    return false;
  }

  ifs.seekg(feature_offset);
  if (!space.load(ifs)) {
    return false;
  }
  space.set_num_labels(labels.size());

  ifs.seekg(parameter_offset);
  if (!param.load(ifs)) {
    return false;
  }

  return true;
}
}     //  end for namespace segmentor
}     //  end for namespace ltp
