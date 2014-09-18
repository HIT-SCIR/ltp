#include "parser/options.h"
#include "parser/featurespace.h"
#include "parser/extractor.h"
#include "parser/treeutils.hpp"
#include "utils/strvec.hpp"
#include "utils/logging.hpp"

namespace ltp {
namespace parser {

using namespace ltp::utility;

int FeatureSpace::retrieve(int gid, int tid, const char * key, bool create) {
  // no boundary check, which is very dangerous
  return groups[gid]->retrieve(tid, key, create);
}

int FeatureSpace::index(int gid, int tid, const char * key, int lid) {
  // no boundary check, which is very dangerous
  int bid = groups[gid]->retrieve(tid, key, false);
  if (bid < 0) return -1;

  return bid * _num_deprels + lid + offsets[gid];
}

void FeatureSpace::build_feature_space_truncate(int num_deprels) {
  _num_deprels = num_deprels;
  allocate_dictionary_groups();
}

void FeatureSpace::set_offset_truncate() {
  _offset=0;
  _num_features=0;
  offsets[DEP]=_offset;
  if(feat_opt.use_dependency) {
    _num_features += groups[DEP]->dim();
    _offset += groups[DEP]->dim() * _num_deprels;
  }

  offsets[SIB]=_offset;
  if(feat_opt.use_sibling) {
    _num_features += groups[SIB]->dim();
    _offset += groups[SIB]->dim() * _num_deprels;
  }

  offsets[GRD]=_offset;
  if(feat_opt.use_grand) {
    _num_features += groups[GRD]->dim();
    _offset += groups[GRD]->dim() * _num_deprels;
  }
}

int FeatureSpace::build_feature_space(int num_deprels,
    const std::vector<Instance *> & instances) {
  _num_deprels = num_deprels;
  // allocate dictionary groups according to the options
  allocate_dictionary_groups();

  // loop over the training instances and extract gold features.
  for (int i = 0; i < instances.size(); ++ i) {
    Instance * inst = instances[i];

    if (feat_opt.use_dependency) {
      int N = DEPExtractor::num_templates();

      for (treeutils::DEPIterator itx(inst->heads); !itx.end(); ++ itx) {
        int hid = itx.hid();
        int cid = itx.cid();

        std::vector< StringVec > cache;
        cache.resize( N );

        DEPExtractor::extract2o(inst, hid, cid, cache);
        for (int k = 0; k < cache.size(); ++ k) {
          for (int itx = 0; itx < cache[k].size(); ++ itx) {
            retrieve(DEP, k, cache[k][itx], true);
          }
        }
      }
    }   //  end for if feat_opt.use_dependency

    if (feat_opt.use_sibling) {
      int N = SIBExtractor::num_templates();

      for (treeutils::SIBIterator itx(inst->heads, feat_opt.use_last_sibling);
           !itx.end();
           ++ itx) {
        int hid = itx.hid();
        int cid = itx.cid();
        int sid = itx.sid();

        std::vector< StringVec > cache;
        cache.resize(N);
        SIBExtractor::extract3o(inst, hid, cid, sid, cache);

        for (int k = 0; k < cache.size(); ++ k) {
          for (int itx = 0; itx < cache[k].size(); ++ itx) {
            retrieve(SIB, k, cache[k][itx], true);
          }
        }
      }
    }   //  end for if feat_opt.use_sibling

    if (feat_opt.use_grand) {
      int N = GRDExtractor::num_templates();

      for (treeutils::GRDIterator itx(inst->heads, feat_opt.use_no_grand);
           !itx.end();
           ++ itx) {
        int hid = itx.hid();
        int cid = itx.cid();
        int gid = itx.gid();

        std::vector< StringVec > cache;
        cache.resize(N);
        GRDExtractor::extract3o(inst, hid, cid, gid, cache);

        for (int k = 0; k < cache.size(); ++ k) {
          for (int itx = 0; itx < cache[k].size(); ++ itx) {
            retrieve(GRD, k, cache[k][itx], true);
          }
        }
      }
    }   //  end for feat_opt.use_grand

    if ((i+1) % model_opt.display_interval== 0) {
      TRACE_LOG("In building feature space, [%d] instances scanned.", i+1);
    }
  }

  _offset = 0;
  _num_features = 0;

  offsets[DEP] = _offset;
  if (feat_opt.use_dependency) {
    _num_features += groups[DEP]->dim();
    _offset += groups[DEP]->dim() * _num_deprels;
  }

  offsets[SIB] = _offset;
  if (feat_opt.use_sibling) {
    _num_features += groups[SIB]->dim();
    _offset += groups[SIB]->dim() * _num_deprels;
  }

  offsets[GRD] = _offset;
  if (feat_opt.use_grand) {
    _num_features += groups[GRD]->dim();
    _offset += groups[GRD]->dim() * _num_deprels;
  }

  /*offsets[GRDSIB] = offset;
  if (feat_opt.use_grandsibling) {
    offset += groups[GRDSIB]->dim() * _num_deprels;
  }*/

  return _offset;
}

int FeatureSpace::allocate_dictionary_groups() {
  int ret = 0;
  if (feat_opt.use_dependency) {
    groups[DEP] = new DictionaryCollections( DEPExtractor::num_templates() );
    ++ ret;
  }

  if (feat_opt.use_sibling) {
    groups[SIB] = new DictionaryCollections( SIBExtractor::num_templates() );
    ++ ret;
  }

  if (feat_opt.use_grand) {
    groups[GRD] = new DictionaryCollections( GRDExtractor::num_templates() );
    ++ ret;
  }

  /*if (feat_opt.use_grand_sibling) {
    groups[GRDSIB] = new DictionaryGroup( GRDSIBExtractor::num_template() );
    ++ ret;
  }

  if (feat_opt.use_postag_unigram) {
    groups[POSU] = new DictionaryGroup( POSUExtractor::num_template() );
    ++ ret;
  }

  if (feat_opt.use_postag_bigram) {
    groups[POSB] = new DictionaryGroup( POSBExtractor::num_template() );
    ++ ret;
  }*/

  return ret;
}

int FeatureSpace::num_features() {
  return _num_features;
}

int FeatureSpace::dim() {
  return _offset;
}

void FeatureSpace::save(std::ostream & out) {
  if (feat_opt.use_dependency) {
    groups[DEP]->dump(out);
  }

  if (feat_opt.use_sibling) {
    groups[SIB]->dump(out);
  }

  if (feat_opt.use_grand) {
    groups[GRD]->dump(out);
  }

  /*if (feat_opt.use_grand_sibling) {
    groups[GRDSIB]->dump(out);
  }

  if (feat_opt.use_postag_unigram) {
    groups[POSU]->dump(out);
  }

  if (feat_opt.use_postag_bigram) {
    groups[POSB]->dump(out);
  }*/
}

bool FeatureSpace::load(int num_deprels, std::istream & in) {

  _num_deprels = num_deprels;
  _offset = 0;
  _num_features = 0;

  offsets[DEP] = _offset;
  if (feat_opt.use_dependency) {
    groups[DEP] = new DictionaryCollections( DEPExtractor::num_templates() );
    if (!groups[DEP]->load(in)) {
      return false;
    }

    _num_features += groups[DEP]->dim();
    _offset += groups[DEP]->dim() * _num_deprels;
  }

  offsets[SIB] = _offset;
  if (feat_opt.use_sibling) {
    groups[SIB] = new DictionaryCollections( SIBExtractor::num_templates() );
    if (!groups[SIB]->load(in)) {
      return false;
    }

    _num_features += groups[SIB]->dim();
    _offset += groups[SIB]->dim() * _num_deprels;
  }

  offsets[GRD] = _offset;
  if (feat_opt.use_grand) {
    groups[GRD] = new DictionaryCollections( GRDExtractor::num_templates() );
    if (!groups[GRD]->load(in)) {
      return false;
    }
    _num_features += groups[GRD]->dim();
    _offset += groups[GRD]->dim() * _num_deprels;
  }

  /*if (feat_opt.use_grand_sibling) {
    groups[GRDSIB] = new DictionaryGroup( GRDSIBExtractor::num_template() );
    ++ ret;
  }

  if (feat_opt.use_postag_unigram) {
    groups[POSU] = new DictionaryGroup( POSUExtractor::num_template() );
    ++ ret;
  }

  if (feat_opt.use_postag_bigram) {
    groups[POSB] = new DictionaryGroup( POSBExtractor::num_template() );
    ++ ret;
  }*/

  return true;

}

}   //  end for namespace parser
}   //  end for namespace ltp
