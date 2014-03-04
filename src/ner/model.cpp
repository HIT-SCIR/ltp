#include "model.h"

namespace ltp {
namespace ner {

Model::Model() {
}

Model::~Model() {
}

void
Model::save(std::ostream & ofs) {
  // write a signature into the file
  char chunk[16] = {'o','t','n','e','r', '\0'};
  ofs.write(chunk, 16);

  int off = ofs.tellp();

  unsigned labels_offset    = 0;
  unsigned lexicon_offset   = 0;
  unsigned feature_offset   = 0;
  unsigned parameter_offset   = 0;

  write_uint(ofs, 0); //  the label offset
  write_uint(ofs, 0); //  the cluster lexicon offset
  write_uint(ofs, 0); //  the features offset
  write_uint(ofs, 0); //  the parameter offset

  labels_offset = ofs.tellp();
  labels.dump(ofs);

  lexicon_offset = ofs.tellp();
  cluster_lexicon.dump(ofs);

  feature_offset = ofs.tellp();
  space.dump(ofs);

  parameter_offset = ofs.tellp();
  param.dump(ofs);

  ofs.seekp(off);
  write_uint(ofs, labels_offset);
  write_uint(ofs, lexicon_offset);
  write_uint(ofs, feature_offset);
  write_uint(ofs, parameter_offset);
}

bool Model::load(std::istream & ifs) {
  char chunk[16];
  ifs.read(chunk, 16);

  if (strcmp(chunk, "otner")) {
    return false;
  }

  unsigned labels_offset  = read_uint(ifs);
  unsigned lexicon_offset   = read_uint(ifs);
  unsigned feature_offset   = read_uint(ifs);
  unsigned parameter_offset = read_uint(ifs);

  ifs.seekg(labels_offset);
  if (!labels.load(ifs)) {
    return false;
  }

  ifs.seekg(lexicon_offset);
  if (!cluster_lexicon.load(ifs)) {
    return false;
  }

  ifs.seekg(feature_offset);
  if (!space.load(labels.size(), ifs)) {
    return false;
  }

  ifs.seekg(parameter_offset);
  if (!param.load(ifs)) {
    return false;
  }

  return true;
}

}     //  end for namespace ner
}     //  end for namespace ltp
