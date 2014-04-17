#include "parser/model.h"

namespace ltp {
namespace parser {

int Model::num_deprels() {
  if (_num_deprels < 0) {
    // unlabeled case
    if (0 == deprels.size()) {
      _num_deprels = 1;
    } else {
      _num_deprels = deprels.size();
    }
  }
  return _num_deprels;
}

int Model::num_postags() {
  if (_num_postags < 0) {
    // unlabeled case
    if (0 == postags.size()) {
      _num_postags = 1;
    } else {
      _num_postags = postags.size();
    }
  }

  return _num_postags;
}

int Model::num_features() {
  if (_num_features < 0) {
    _num_features = space.num_features();
  }
  return _num_features;
}

int Model::dim() {
  if (_dim < 0) {
    _dim = space.dim();
  }
  return _dim;
}

void Model::save(ostream & out) {
  // write a signature
  char chunk[16] = {'l','g','d','p', 'j', 0};
  out.write(chunk, 16);
  unsigned int tmp;

  int off = out.tellp();

  unsigned basic_offset =  0;
  unsigned postag_offset = 0;
  unsigned deprels_offset = 0;
  unsigned feature_offset = 0;
  unsigned parameter_offset = 0;

  // write pseduo position
  write_uint(out, 0); //  basic offset
  write_uint(out, 0); //  postag offset
  write_uint(out, 0); //  deprels offset
  write_uint(out, 0); //  features offset
  write_uint(out, 0); //  parameters offset

  // model and feature information
  // labeled model
  basic_offset = out.tellp();
  tmp = model_opt.labeled;
  write_uint(out, tmp);

  // decode order
  strncpy(chunk, model_opt.decoder_name.c_str(), 16);
  out.write(chunk, 16);

  // use dependency
  tmp = feat_opt.use_dependency;
  write_uint(out, tmp);

  // use dependency unigram
  tmp = feat_opt.use_dependency_unigram;
  write_uint(out, tmp);

  // use dependency bigram
  tmp = feat_opt.use_dependency_bigram;
  write_uint(out, tmp);

  // use dependency surrounding
  tmp = feat_opt.use_dependency_surrounding;
  write_uint(out, tmp);

  // use dependency between
  tmp = feat_opt.use_dependency_between;
  write_uint(out, tmp);

  // use sibling
  tmp = feat_opt.use_sibling;
  write_uint(out, tmp);

  // use sibling basic
  tmp = feat_opt.use_sibling_basic;
  write_uint(out, tmp);

  // use sibling linear
  tmp = feat_opt.use_sibling_linear;
  write_uint(out, tmp);

  // use grand
  tmp = feat_opt.use_grand;
  write_uint(out, tmp);

  // use grand basic
  tmp = feat_opt.use_grand_basic;
  write_uint(out, tmp);

  // use grand linear
  tmp = feat_opt.use_grand_linear;
  write_uint(out, tmp);

  // save postag lexicon
  postag_offset = out.tellp();
  postags.dump(out);

  // save dependency relation lexicon
  deprels_offset = out.tellp();
  deprels.dump(out);

  feature_offset = out.tellp();
  space.save(out);

  parameter_offset = out.tellp();
  param.dump(out);

  out.seekp(off);
  write_uint(out, basic_offset);
  write_uint(out, postag_offset);
  write_uint(out, deprels_offset);
  write_uint(out, feature_offset);
  write_uint(out, parameter_offset);

  // out.seekp(0, std::ios::end);
}

bool Model::load(istream & in) {
  char chunk[16];
  in.read(chunk, 16);
  if (strcmp(chunk, "lgdpj")) {
    return false;
  }

  unsigned int basic_offset = read_uint(in);
  unsigned int postag_offset = read_uint(in);
  unsigned int deprels_offset = read_uint(in);
  unsigned int feature_offset = read_uint(in);
  unsigned int parameter_offset = read_uint(in);

  in.seekg(basic_offset);
  model_opt.labeled = (read_uint(in) == 1);

  // decode order
  in.read(chunk, 16);
  model_opt.decoder_name = chunk;

  // use dependency
  feat_opt.use_dependency = (read_uint(in) == 1);

  // use dependency unigram
  feat_opt.use_dependency_unigram = (read_uint(in) == 1);

  // use dependency bigram
  feat_opt.use_dependency_bigram = (read_uint(in) == 1);

  // use dependency surrounding
  feat_opt.use_dependency_surrounding = (read_uint(in) == 1);

  // use dependency between
  feat_opt.use_dependency_between = (read_uint(in) == 1);

  // use sibling
  feat_opt.use_sibling = (read_uint(in) == 1);

  // use sibling basic
  feat_opt.use_sibling_basic = (read_uint(in) == 1);

  // use sibling linear
  feat_opt.use_sibling_linear = (read_uint(in) == 1);

  // use grand
  feat_opt.use_grand = (read_uint(in) == 1);

  // use grand basic
  feat_opt.use_grand_basic = (read_uint(in) == 1);

  // use grand linear
  feat_opt.use_grand_linear = (read_uint(in) == 1);

  // automically detrieve
  feat_opt.use_unlabeled_dependency = (!model_opt.labeled
                                       && feat_opt.use_dependency);

  feat_opt.use_labeled_dependency = (model_opt.labeled
                                     && feat_opt.use_dependency);

  feat_opt.use_unlabeled_sibling = (!model_opt.labeled
                                    && feat_opt.use_sibling);

  feat_opt.use_labeled_sibling = (model_opt.labeled
                                  && feat_opt.use_sibling);

  feat_opt.use_unlabeled_grand = (!model_opt.labeled
                                  && feat_opt.use_grand);

  feat_opt.use_labeled_grand = (model_opt.labeled
                                && feat_opt.use_grand);

  in.seekg(postag_offset);
  if (!postags.load(in)) {
    return false;
  }

  in.seekg(deprels_offset);
  if (!deprels.load(in)) {
    return false;
  }

  in.seekg(feature_offset);
  if (!space.load(num_deprels(), in)) {
    return false;
  }

  in.seekg(parameter_offset);
  if (!param.load(in)) {
    return false;
  }

  return true;
}

}     //  end for namespace parser

}     //  end for namespace ltp
