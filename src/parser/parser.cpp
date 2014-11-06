#include "parser/parser.h"
#include "parser/options.h"
#include "parser/decoder1o.h"
#include "parser/decoder2o.h"
#include "parser/conllreader.h"
#include "parser/conllwriter.h"
#include "parser/treeutils.hpp"

#if _WIN32
#include <Windows.h>
#define sleep Sleep
#endif  //  end for _WIN32

namespace ltp {
namespace parser {

Parser::Parser() :
  __TRAIN__(false),
  __TEST__(false),
  model(0),
  decoder(0) {
  init_opt();
}

Parser::Parser(ConfigParser & cfg) :
  __TRAIN__(false),
  __TEST__(false),
  model(0),
  decoder(0) {
  init_opt();
  parse_cfg(cfg);
}

Parser::~Parser() {
  if (decoder) {
    delete decoder;
  }

  if (model) {
    delete model;
  }

  for (int i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) {
      delete train_dat[i];
    }
  }
}

void
Parser::init_opt() {
  model_opt.labeled           = false;
  model_opt.decoder_name      = "1o";
  model_opt.display_interval  = 1000;

  train_opt.train_file              = "";
  train_opt.holdout_file            = "";
  train_opt.max_iter                = 10;
  train_opt.algorithm               = "pa";
  train_opt.model_name              = "";
  train_opt.rare_feature_threshold  = 0;

  test_opt.test_file  = "";
  test_opt.model_file = "";

  feat_opt.use_postag                 = false;
  feat_opt.use_postag_unigram         = false;
  feat_opt.use_postag_bigram          = false;

  feat_opt.use_dependency             = false;
  feat_opt.use_dependency_unigram     = false;
  feat_opt.use_dependency_bigram      = false;
  feat_opt.use_dependency_surrounding = false;
  feat_opt.use_dependency_between     = false;

  feat_opt.use_sibling                = false;
  feat_opt.use_sibling_basic          = false;
  feat_opt.use_sibling_linear         = false;

  feat_opt.use_grand                  = false;
  feat_opt.use_grand_basic            = false;
  feat_opt.use_grand_linear           = false;

  feat_opt.use_last_sibling           = false;
  feat_opt.use_no_grand               = false;
  feat_opt.use_distance_in_features   = true;
}

Model *
Parser::erase_rare_features(const int * feature_group_updated_time) {
  Model * new_model = new Model;

  for(int i = 0; i < model->deprels.size(); ++ i) {
    const char * key = model-> deprels.at(i);
    new_model->deprels.push(key);
  }

  for(int i = 0; i< model->postags.size(); ++ i) {
    const char * key = model -> postags.at(i);
    new_model->postags.push(key);
  }

  build_feature_space_truncate(new_model);

  if(feat_opt.use_dependency) {//DEP
    copy_featurespace(new_model,
                      FeatureSpace::DEP,
                      feature_group_updated_time);
  }

  if(feat_opt.use_sibling) {//SIB
    copy_featurespace(new_model,
                      FeatureSpace::SIB,
                      feature_group_updated_time);
  }

  if(feat_opt.use_grand) {//GRD
    copy_featurespace(new_model,
                      FeatureSpace::GRD,
                      feature_group_updated_time);
  }

  TRACE_LOG("Scanning old features space, building new feature space is done");

  new_model->space.set_offset_truncate();
  TRACE_LOG("Setting offset for each collection is done");

  new_model->param.realloc(new_model->dim());
  TRACE_LOG("Parameter dimension of new model is [%d]",new_model->space.dim());

  if(feat_opt.use_dependency) {//DEP
    copy_parameters(new_model, FeatureSpace::DEP);
  }

  if(feat_opt.use_sibling) {//SIB
    copy_parameters(new_model, FeatureSpace::SIB);
  }

  if(feat_opt.use_grand) {//GRD
    copy_parameters(new_model, FeatureSpace::GRD);
  }

  TRACE_LOG("Building new model is done");
  return new_model;
}

void
Parser::copy_featurespace(Model * new_model,
                          int gid,
                          const int * feature_group_updated_time) {
  // perform the feature space truncation
  // the stategy here is travel through the old feature space, test a group of
  // feature whether their are all zero. if all zero detected, don't insert it
  // into new feature space.

  int L = model->num_deprels();
  for (FeatureSpaceIterator itx = model->space.begin(gid); !itx.end(); ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(gid, tid, key);
    bool flag = false;

    for (int l = 0; l < L; ++ l) {
      double p = model -> param.dot(id+l);
      if(p != 0.) {
        flag=true;
      }
    }

    if(!flag) {
      continue;
    }

    int idx = id / L;
    if(feature_group_updated_time
       && (feature_group_updated_time[idx] < train_opt.rare_feature_threshold)) {
      continue;
    }

    new_model->space.retrieve(gid, tid, key, true);
  }
}

void
Parser::copy_parameters(Model * new_model, int gid) {
  // perform the parameter trunction
  // the prerequiest is feature space of new model is already built.
  // the process travel through the feature space of new model and retrieve
  // the key in old feature space, then preform the copy operation.
  int L = model-> num_deprels();

  for (FeatureSpaceIterator itx = new_model->space.begin(gid); !itx.end(); ++itx) {
    const char * key = itx.key();
    int tid = itx.tid();

    int old_id = model->space.index(gid, tid, key);
    int new_id = new_model->space.index(gid, tid, key);

    for (int l = 0; l < L; ++l) {
      new_model->param._W[new_id + l]     = model->param._W[old_id+l];
      new_model->param._W_sum[new_id + l] = model->param._W_sum[old_id+l];
    }
  }
}

bool
Parser::parse_cfg(utility::ConfigParser & cfg) {
  string  strbuf;
  int   intbuf;

  if (cfg.has_section("model")) {
    if (cfg.get_integer("model", "labeled", intbuf)) {
      model_opt.labeled = (intbuf == 1);
    }
    if (cfg.get("model", "decoder-name", strbuf)) {
      model_opt.decoder_name = strbuf;
    } else {
      WARNING_LOG("decoder-name is not configed, [1o] is set as default.");
    }
  }

  __TRAIN__ = false;

  if (cfg.has_section("train")) {
    TRACE_LOG("train model specified.");
    __TRAIN__ = true;

    if (cfg.get("train", "train-file", strbuf)) {
      train_opt.train_file = strbuf;
    } else {
      ERROR_LOG("train-file config item is not set.");
      return false;
    }   //  end for if (cfg.get("train", "train-file", strbuf))

    if (cfg.get("train", "holdout-file", strbuf)) {
      train_opt.holdout_file = strbuf;
    } else {
      ERROR_LOG("holdout-file config item is not set.");
      return false;
    }

    if (cfg.get("train", "algorithm", strbuf)) {
      train_opt.algorithm = strbuf;
    } else {
      WARNING_LOG("algorithm is not configed, [PA] is set as default.");
    }

    train_opt.model_name = (train_opt.train_file
                            + "."
                            + train_opt.algorithm
                            + ".model");
    if (cfg.get("train", "model-name", strbuf)) {
      train_opt.model_name = strbuf;
    } else {
      WARNING_LOG("model name is not configed, [%s] is set as default",
                  train_opt.model_name.c_str());
    }

    if (cfg.get_integer("train", "rare-feature-threshold", intbuf)) {
      train_opt.rare_feature_threshold = intbuf;
    } else {
      WARNING_LOG("rare feature's threshold is not configed, use 0 as default");
    }

    if (cfg.get_integer("train", "max-iter", intbuf)) {
      train_opt.max_iter = intbuf;
    } else {
      WARNING_LOG("max-iter is not configed, [10] is set as default.");
    }
  }   //  end for cfg.has_section("train")

  __TEST__ = false;

  if (cfg.has_section("test")) {
    __TEST__ = true;

    if (cfg.get("test", "test-file", strbuf)) {
      test_opt.test_file = strbuf;
    } else {
      ERROR_LOG("test-file config item is not set.");
      return false;
    }   //  end for if (cfg.get("train", "train-file", strbuf))

    if (cfg.get("test", "model-file", strbuf)) {
      test_opt.model_file = strbuf;
    } else {
      ERROR_LOG("model-file config item is not set.");
      return false;
    }
  }

  if (cfg.has_section("feature")) {
    if (cfg.get_integer("feature", "use-postag", intbuf)) {
      feat_opt.use_postag = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-postag-unigram", intbuf)) {
      feat_opt.use_postag_unigram = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-postag-bigram", intbuf)) {
      feat_opt.use_postag_bigram = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-dependency", intbuf)) {
      feat_opt.use_dependency = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-dependency-unigram", intbuf)) {
      feat_opt.use_dependency_unigram = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-dependency-bigram", intbuf)) {
      feat_opt.use_dependency_bigram = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-dependency-surrounding", intbuf)) {
      feat_opt.use_dependency_surrounding = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-dependency-between", intbuf)) {
      feat_opt.use_dependency_between = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-sibling", intbuf)) {
      feat_opt.use_sibling = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-sibling-basic", intbuf)) {
      feat_opt.use_sibling_basic = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-sibling-linear", intbuf)) {
      feat_opt.use_sibling_linear = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-grand", intbuf)) {
      feat_opt.use_grand = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-grand-basic", intbuf)) {
      feat_opt.use_grand_basic = (intbuf == 1);
    }

    if (cfg.get_integer("feature", "use-grand-linear", intbuf)) {
      feat_opt.use_grand_linear = (intbuf == 1);
    }
  }

  // incompatible configuration check
  if (model_opt.decoder_name == "1o") {
    if (feat_opt.use_sibling) {
      WARNING_LOG("Sibling features should not be configed "
                  "with 1st-order decoder.");
      TRACE_LOG("Sibling features is inactived.");
      feat_opt.use_sibling = false;
    }

    if (feat_opt.use_grand) {
      WARNING_LOG("Grandchild features should not be configed "
                  "with 1st-order decoder.");
      TRACE_LOG("Grandchild features is inactived.");
      feat_opt.use_grand = false;
    }
  } else if (model_opt.decoder_name == "2o-sib") {
    if (feat_opt.use_grand) {
      WARNING_LOG("Grandchild features should not be configed "
                   "with 2nd-order-sibling decoder.");
      TRACE_LOG("Grandchild features is inactived.");
      feat_opt.use_grand = false;
    }
  }

  // detrieve dependency type from configuration
  feat_opt.use_unlabeled_dependency = (model_opt.labeled == false
                                       && feat_opt.use_dependency);
  feat_opt.use_labeled_dependency   = (model_opt.labeled == true
                                       && feat_opt.use_dependency);

  // detrieve sibling type from configuration
  feat_opt.use_unlabeled_sibling    = (model_opt.labeled == false
                                       && feat_opt.use_sibling);
  feat_opt.use_labeled_sibling      = (model_opt.labeled == true
                                       && feat_opt.use_sibling);
  feat_opt.use_unlabeled_grand      = (model_opt.labeled == false
                                       && feat_opt.use_grand);
  feat_opt.use_labeled_grand        = (model_opt.labeled == true
                                       && feat_opt.use_grand);

  return true;
}

void
Parser::build_configuration(void) {
  // build postags set, deprels set.
  // map deprels from string to int when model_opt.labeled is configed.
  // need to check if the model is initialized

  for (int i = 0; i < train_dat.size(); ++ i) {
    int len = train_dat[i]->size();

    // if labeled is configured, init the deprelsidxs
    if (model_opt.labeled) {
      train_dat[i]->deprelsidx.resize(len);
      train_dat[i]->predicted_deprelsidx.resize(len);
    }

    for (int j = 1; j < len; ++ j) {
      model->postags.push(train_dat[i]->postags[j].c_str());
      if (model_opt.labeled) {
        int idx = model->deprels.push(train_dat[i]->deprels[j].c_str());
        train_dat[i]->deprelsidx[j] = idx;
      }
    }
  }
}

void
Parser::build_feature_space(void) {
  model->space.build_feature_space(model->num_deprels(), train_dat);
}   //  end for build_feature_space

void
Parser::build_feature_space_truncate(Model * m) {
  m->space.build_feature_space_truncate(m->num_deprels());
}

void
Parser::collect_unlabeled_features_of_one_instance(Instance * inst,
                                                   const vector<int> & heads,
                                                   SparseVec & vec ) {

  vec.zero();
  if (feat_opt.use_dependency) {
    for (treeutils::DEPIterator itx(heads); !itx.end(); ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();

      const FeatureVector * fv = inst->depu_features[hid][cid];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, 1.);
    }
  }

  if (feat_opt.use_sibling) {
    for (treeutils::SIBIterator itx(heads, feat_opt.use_last_sibling);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int sid = itx.sid();

      const FeatureVector * fv = inst->sibu_features[hid][cid][sid];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, 1.);
    }
  }

  if (feat_opt.use_grand) {
    for (treeutils::GRDIterator itx(heads, feat_opt.use_no_grand);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int gid = itx.gid();

      const FeatureVector * fv = inst->grdu_features[hid][cid][gid];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, 1.);
    }
  }
}

void
Parser::collect_labeled_features_of_one_instance(Instance * inst,
                                                 const vector<int> & heads,
                                                 const vector<int> & deprelsidx,
                                                 SparseVec & vec) {
  vec.zero();
  if (feat_opt.use_dependency) {
    for (treeutils::DEPIterator itx(heads); !itx.end(); ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int relidx = deprelsidx[cid];

      const FeatureVector * fv = inst->depl_features[hid][cid][relidx];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);
    }
  }

  if (feat_opt.use_sibling) {
    for (treeutils::SIBIterator itx(heads, feat_opt.use_last_sibling);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int sid = itx.sid();
      int relidx = deprelsidx[cid];

      const FeatureVector * fv = inst->sibl_features[hid][cid][sid][relidx];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);
    }
  }

  if (feat_opt.use_grand) {
    for (treeutils::GRDIterator itx(heads, feat_opt.use_no_grand);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int gid = itx.gid();
      int relidx = deprelsidx[cid];

      const FeatureVector * fv = inst->grdl_features[hid][cid][gid][relidx];
      if (NULL == fv) {
        continue;
      }

      vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);
    }
  }
}

void
Parser::collect_features_of_one_instance(Instance * inst, bool gold) {
  if (gold) {
    if (!model_opt.labeled) {
      collect_unlabeled_features_of_one_instance(inst,
          inst->heads,
          inst->features);
    } else {
      collect_labeled_features_of_one_instance(inst,
          inst->heads,
          inst->deprelsidx,
          inst->features);
    }
  } else {
    if (!model_opt.labeled) {
      collect_unlabeled_features_of_one_instance(inst,
          inst->predicted_heads,
          inst->predicted_features);
    } else {
      collect_labeled_features_of_one_instance(inst,
          inst->predicted_heads,
          inst->predicted_deprelsidx,
          inst->predicted_features);
    }
  }
}

bool
Parser::read_instances(const char * filename, vector<Instance *> & dat) {
  Instance * inst = NULL;
  ifstream f(filename);
  if (!f) {
    return false;
  }

  CoNLLReader reader(f);

  int num_inst = 0;

  while ((inst = reader.next())) {
    dat.push_back(inst);
    ++ num_inst;

    if (num_inst % model_opt.display_interval == 0) {
      TRACE_LOG("Reading in [%d] instances.", num_inst);
    }
  }

  return true;
}

Decoder *
Parser::build_decoder(void) {
  Decoder * deco = NULL;
  if (model_opt.decoder_name == "1o") {
    if (!model_opt.labeled) {
      deco = new Decoder1O();
    } else {
      deco = new Decoder1O(model->num_deprels());
    }

  } else if (model_opt.decoder_name == "2o-sib") {
    if (!model_opt.labeled) {
      deco = new Decoder2O();
    } else {
      deco = new Decoder2O(model->num_deprels());
    }

  } else if (model_opt.decoder_name == "2o-carreras") {
    if (!model_opt.labeled) {
      deco = new Decoder2OCarreras();
    } else {
      deco = new Decoder2OCarreras(model->num_deprels());
    }
  }
  return deco;
}

void
Parser::extract_features(Instance * inst) {
  int len = inst->size();
  int L   = model->num_deprels();
  // FeatureSpace & space = model->space;

  if (feat_opt.use_dependency) {

    if (!model_opt.labeled) {
      inst->depu_features.resize(len, len);
      inst->depu_scores.resize(len, len);

      inst->depu_features = 0;
      inst->depu_scores = DOUBLE_NEG_INF;
    } else {
      inst->depl_features.resize(len, len, L);
      inst->depl_scores.resize(len, len, L);

      inst->depl_features = 0;
      inst->depl_scores = DOUBLE_NEG_INF;
    }

    vector< StringVec >  cache;
    vector< int >   cache_again;

    int N = DEPExtractor::num_templates();

    cache.resize( N );

    for (treeutils::DEPTreeSpaceIterator itx(len); !itx.end(); ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();

      // here the self-implementated String Vector is little
      // fasteer than the list<string>
      for (int i = 0; i < N; ++ i) {
        cache[i].clear();
      }

      DEPExtractor::extract2o(inst, hid, cid, cache);
      cache_again.clear();

      for (int tid = 0; tid < cache.size(); ++ tid) {
        for (int itx = 0; itx < cache[tid].size(); ++ itx) {
          int idx = model->space.index(FeatureSpace::DEP, tid, cache[tid][itx], 0);
          // std::cout << "idx: " << idx << std::endl;
          if (idx >= 0) {
            cache_again.push_back(idx);
          }
        }
      }

      int num_feat = cache_again.size();

      if (num_feat > 0) {
        if (!model_opt.labeled) {
          inst->depu_features[hid][cid] = new FeatureVector;
          inst->depu_features[hid][cid]->n = num_feat;
          inst->depu_features[hid][cid]->idx = 0;
          inst->depu_features[hid][cid]->val = 0;

          inst->depu_features[hid][cid]->idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            inst->depu_features[hid][cid]->idx[j] = cache_again[j];
          }
        } else {
          int l = 0;
          int * idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            idx[j] = cache_again[j];
          }

          inst->depl_features[hid][cid][l] = new FeatureVector;
          inst->depl_features[hid][cid][l]->n = num_feat;
          inst->depl_features[hid][cid][l]->val = 0;
          inst->depl_features[hid][cid][l]->loff = 0;
          inst->depl_features[hid][cid][l]->idx = idx;

          for (l = 1; l < L; ++ l) {
            inst->depl_features[hid][cid][l] = new FeatureVector;
            inst->depl_features[hid][cid][l]->n = num_feat;
            inst->depl_features[hid][cid][l]->idx = idx;
            inst->depl_features[hid][cid][l]->val = 0;
            inst->depl_features[hid][cid][l]->loff = l;
          }
        }
      }
    }   //  end for DEPTreeSpaceIterator itx
  }   //  end for feat_opt.use_dependency

  if (feat_opt.use_sibling) {
    if (!model_opt.labeled) {
      inst->sibu_features.resize(len, len, len);
      inst->sibu_scores.resize(len, len, len);

      inst->sibu_features = 0;
      inst->sibu_scores = DOUBLE_NEG_INF;
    } else {
      inst->sibl_features.resize(len, len, len, L);
      inst->sibl_scores.resize(len, len, len, L);

      inst->sibl_features = 0;
      inst->sibl_scores = DOUBLE_NEG_INF;
    }

    int N = SIBExtractor::num_templates();

    vector< StringVec > cache;
    vector< int > cache_again;

    cache.resize(N);

    for (treeutils::SIBTreeSpaceIterator itx(len, feat_opt.use_last_sibling);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int sid = itx.sid();

      for (int i = 0; i < N; ++ i) {
        cache[i].clear();
      }

      SIBExtractor::extract3o(inst, hid, cid, sid, cache);
      cache_again.clear();

      //
      for (int tid = 0; tid < cache.size(); ++ tid) {
        for (int itx = 0; itx < cache[tid].size(); ++ itx) {
          int idx = model->space.index(FeatureSpace::SIB, tid, cache[tid][itx]);
          if (idx >= 0) {
            cache_again.push_back(idx);
          }
        }
      }

      int num_feat = cache_again.size();

      if (num_feat > 0) {
        if (!model_opt.labeled) {
          inst->sibu_features[hid][cid][sid] = new FeatureVector;
          inst->sibu_features[hid][cid][sid]->n = num_feat;
          inst->sibu_features[hid][cid][sid]->idx = 0;
          inst->sibu_features[hid][cid][sid]->val = 0;

          inst->sibu_features[hid][cid][sid]->idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            inst->sibu_features[hid][cid][sid]->idx[j] = cache_again[j];
          }
        } else {
          int l = 0;
          int * idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            idx[j] = cache_again[j];
          }

          inst->sibl_features[hid][cid][sid][l] = new FeatureVector;
          inst->sibl_features[hid][cid][sid][l]->n = num_feat;
          inst->sibl_features[hid][cid][sid][l]->val = 0;
          inst->sibl_features[hid][cid][sid][l]->idx = idx;
          inst->sibl_features[hid][cid][sid][l]->loff = 0;

          for (l = 1; l < L; ++ l) {
            inst->sibl_features[hid][cid][sid][l] = new FeatureVector;
            inst->sibl_features[hid][cid][sid][l]->n = num_feat;
            inst->sibl_features[hid][cid][sid][l]->val = 0;
            inst->sibl_features[hid][cid][sid][l]->idx = idx;
            inst->sibl_features[hid][cid][sid][l]->loff = l;
          }   //  end for if model_opt.labeled
        }
      }
    }   //  end for SIBTreeSpaceIterator itx
  }   //  end for feat_opt.use_sibling

  if (feat_opt.use_grand) {
    if (!model_opt.labeled) {
      inst->grdu_features.resize(len, len, len);
      inst->grdu_scores.resize(len, len, len);

      inst->grdu_features = 0;
      inst->grdu_scores = DOUBLE_NEG_INF;
    } else {
      inst->grdl_features.resize(len, len, len, L);
      inst->grdl_scores.resize(len, len, len, L);

      inst->grdl_features = 0;
      inst->grdl_scores = DOUBLE_NEG_INF;
    }

    int N = GRDExtractor::num_templates();

    vector< StringVec > cache;
    vector< int > cache_again;

    cache.resize(N);

    for (treeutils::GRDTreeSpaceIterator itx(len, feat_opt.use_no_grand);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int gid = itx.gid();

      for (int i = 0; i < N; ++ i) {
        cache[i].clear();
      }

      GRDExtractor::extract3o(inst, hid, cid, gid, cache);
      cache_again.clear();

      for (int tid = 0; tid < cache.size(); ++ tid) {
        for (int itx = 0; itx < cache[tid].size(); ++ itx) {
          int idx = model->space.index(FeatureSpace::GRD, tid, cache[tid][itx]);
          if (idx >= 0) {
            cache_again.push_back(idx);
          }
        }
      }

      int num_feat = cache_again.size();

      if (num_feat > 0) {
        if (!model_opt.labeled) {
          inst->grdu_features[hid][cid][gid] = new FeatureVector;
          inst->grdu_features[hid][cid][gid]->n = num_feat;
          inst->grdu_features[hid][cid][gid]->idx = 0;
          inst->grdu_features[hid][cid][gid]->val = 0;

          inst->grdu_features[hid][cid][gid]->idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            inst->grdu_features[hid][cid][gid]->idx[j] = cache_again[j];
          }
        } else {
          int l = 0;
          int * idx = new int[num_feat];
          for (int j = 0; j < num_feat; ++ j) {
            idx[j] = cache_again[j];
          }

          inst->grdl_features[hid][cid][gid][l] = new FeatureVector;
          inst->grdl_features[hid][cid][gid][l]->n = num_feat;
          inst->grdl_features[hid][cid][gid][l]->val = 0;
          inst->grdl_features[hid][cid][gid][l]->idx = idx;
          inst->grdl_features[hid][cid][gid][l]->loff = 0;

          for (l = 1; l < L; ++ l) {
            inst->grdl_features[hid][cid][gid][l] = new FeatureVector;
            inst->grdl_features[hid][cid][gid][l]->n = num_feat;
            inst->grdl_features[hid][cid][gid][l]->val = 0;
            inst->grdl_features[hid][cid][gid][l]->idx = idx;
            inst->grdl_features[hid][cid][gid][l]->loff = l;
          }   //  end for if model_opt.labeled
        }
      }
    }
  }   //  end for feat_opt.use_grand
}

void
Parser::extract_features(vector<Instance *>& dat) {
  // ofstream fout("lgdpj.fv.tmp", std::ofstream::binary);
  // DependencyExtractor
  for (int i = 0; i < dat.size(); ++ i) {
    extract_features(dat[i]);

    // dat[i]->dump_all_featurevec(fout);
    if ((i + 1) % model_opt.display_interval == 0) {
      TRACE_LOG("[%d] instance is extracted.", i + 1);
    }
  }   // end for i = 0; i < dat.size(); ++ i

  // fout.close();
}

void
Parser::build_gold_features() {
  // ifstream fin("lgdpj.fv.tmp", std::ifstream::binary);
  for (int i = 0; i < train_dat.size(); ++ i) {
    // train_dat[i]->load_all_featurevec(fin);
    collect_features_of_one_instance(train_dat[i], true);
    // train_dat[i]->nice_all_featurevec();
  }
  // fin.close();
}

void
Parser::increase_group_updated_time(const math::SparseVec & vec,
                                    int * feature_group_updated_time) {
  if (!feature_group_updated_time) {
    return;
  }

  int L = model->num_deprels();
  for (math::SparseVec::const_iterator itx = vec.begin();
      itx != vec.end();
      ++ itx) {

    int idx = itx->first;
    if (itx->second != 0.0) {
      ++ feature_group_updated_time[idx / L];
    }
  }
}


void
Parser::train(void) {
  const char * train_file   = train_opt.train_file.c_str();
  // const char * holdout_file = train_opt.holdout_file.c_str();

  if (!read_instances(train_file, train_dat)) {
    ERROR_LOG("Failed to read train data from [%s].", train_file);
    return;
  } else {
    TRACE_LOG("Read in [%d] train instances.", train_dat.size());
  }

  model = new Model;

  TRACE_LOG("Start building configuration.");
  build_configuration();
  TRACE_LOG("Building configuration is done.");
  TRACE_LOG("Number of postags: [%d]", model->num_postags());
  TRACE_LOG("Number of deprels: [%d]", model->num_deprels());

  TRACE_LOG("Start building feature space.");
  build_feature_space();
  TRACE_LOG("Building feature space is done.");
  TRACE_LOG("Number of features: [%d]", model->space.num_features());

  model->param.realloc(model->dim());
  TRACE_LOG("Allocate a parameter vector of [%d] dimension.", model->dim());

  int nr_feature_groups = model->num_features();
  // int num_l = model->num_deprels();
  int * feature_group_updated_time = NULL;

  if (model_opt.labeled
      && train_opt.rare_feature_threshold > 0) {
    feature_group_updated_time = new int[nr_feature_groups];
    for (int i = 0; i < nr_feature_groups; ++ i) {
      feature_group_updated_time[i] = 0;
    }
  }

  decoder = build_decoder();

  int best_iteration = -1;
  double best_las = -1;
  double best_uas = -1;

  for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
    TRACE_LOG("Start training epoch #%d.", (iter + 1));

    // random_shuffle(train_dat.begin(), train_dat.end());
    for (int i = 0; i < train_dat.size(); ++ i) {

      extract_features(train_dat[i]);
      calculate_score(train_dat[i], model->param);
      decoder->decode(train_dat[i]);
      collect_features_of_one_instance(train_dat[i], true);
      collect_features_of_one_instance(train_dat[i], false);

      // instance_verify(train_dat[i], cout, true);

      if (train_opt.algorithm == "pa") {
        SparseVec update_features;
        update_features.zero();
        update_features.add(train_dat[i]->features, 1.);
        update_features.add(train_dat[i]->predicted_features, -1.);
        increase_group_updated_time(update_features,
                                    feature_group_updated_time);

        double error = train_dat[i]->num_errors();
        double score = model->param.dot(update_features, false);
        double norm = update_features.L2();
        double step = 0.;

        if (norm < EPS) {
          step = 0;
        } else {
          step = (error - score) / norm;
        }

        model->param.add(update_features,
                         iter * train_dat.size() + i + 1,
                         step);
      } else if (train_opt.algorithm == "ap") {
        SparseVec update_features;

        update_features.add(train_dat[i]->features, 1.);
        update_features.add(train_dat[i]->predicted_features, -1.);
        increase_group_updated_time(update_features,
                                    feature_group_updated_time);

        model->param.add(update_features,
                         iter * train_dat.size() + i + 1,
                         1.);
      }

      if ((i + 1) % model_opt.display_interval == 0) {
        TRACE_LOG("[%d] instances is trained.", i + 1);
      }

      train_dat[i]->cleanup();
    }

    model->param.flush( train_dat.size() * (iter + 1) );

    Model * new_model;

    new_model = erase_rare_features(feature_group_updated_time);
    swap(model,new_model);

    double las, uas;
    evaluate(las, uas);

    if(las > best_las) {
      best_las = las;
      best_uas = uas;
      best_iteration = iter;
    }

    string saved_model_file = (train_opt.model_name
                               + "."
                               + to_str(iter)
                               + ".model");
    ofstream fout(saved_model_file.c_str(), std::ofstream::binary);

    swap(model,new_model);
    new_model->save(fout);
    delete new_model;

    TRACE_LOG("Model for iteration [%d] is saved to [%s]",
        iter + 1,
        saved_model_file.c_str());

  }

  if (feature_group_updated_time) {
    delete [](feature_group_updated_time);
  }
  TRACE_LOG("Best result (iteration = %d) : LAS = %lf | UAS = %f",
            best_iteration,
            best_las,
            best_uas);

  delete model;
  model = 0;
}

void
Parser::evaluate(double &las, double &uas) {
  const char * holdout_file = train_opt.holdout_file.c_str();

  int head_correct = 0;
  int label_correct = 0;
  int total_rels = 0;

  ifstream f(holdout_file);
  CoNLLReader reader(f);

  Instance * inst = NULL;

  double before = get_time();
  while ((inst = reader.next())) {

    if (model_opt.labeled) {
      inst->deprelsidx.resize(inst->size());
      for (int i = 1; i < inst->size(); ++ i) {
        inst->deprelsidx[i] = model->deprels.index(inst->deprels[i].c_str());
      }
    }

    extract_features(inst);
    calculate_score(inst, model->param, true);

    decoder->decode(inst);

    total_rels += inst->num_rels();
    head_correct += inst->num_correct_heads();
    label_correct += inst->num_correct_heads_and_labels();

    delete inst;
  }

  uas = (double)head_correct / total_rels;
  TRACE_LOG("UAS: %.4lf ( %d / %d )", uas,
                                      head_correct,
                                      total_rels);

  las = 0;
  if (model_opt.labeled) {
    las = (double)label_correct / total_rels;
    TRACE_LOG("LAS: %.4lf ( %d / %d )", las,
                                        label_correct,
                                        total_rels);
  }

  double after = get_time();
  TRACE_LOG("consuming time: %.2lf", after - before);

  // holdout_dat.clear();
}

void
Parser::test() {
  double before = get_time();
  const char * model_file = test_opt.model_file.c_str();
  ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to open file [%s].", model_file);
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  TRACE_LOG("Number of postags        [%d]", model->num_postags());
  TRACE_LOG("Number of deprels        [%d]", model->num_deprels());
  TRACE_LOG("Number of features       [%d]", model->num_features());
  TRACE_LOG("Number of dimension      [%d]", model->dim());
  TRACE_LOG("Labeled:                         %s",
            (model_opt.labeled ? "true" : "fales"));
  TRACE_LOG("Decoder:                         %s",
            model_opt.decoder_name.c_str());
  TRACE_LOG("Dependency features:             %s",
            (feat_opt.use_dependency ?             "true" : "false"));
  TRACE_LOG("Dependency features unigram:     %s",
            (feat_opt.use_dependency_unigram ?     "true" : "false"));
  TRACE_LOG("Dependency features bigram:      %s",
            (feat_opt.use_dependency_bigram ?      "true" : "false"));
  TRACE_LOG("Dependency features surrounding: %s",
            (feat_opt.use_dependency_surrounding ? "true" : "false"));
  TRACE_LOG("Dependency features between:     %s",
            (feat_opt.use_dependency_between ?     "true" : "false"));
  TRACE_LOG("Sibling features:                %s",
            (feat_opt.use_sibling ?                "true" : "false"));
  TRACE_LOG("Sibling basic features:          %s",
            (feat_opt.use_sibling_basic ?          "true" : "false"));
  TRACE_LOG("Sibling linear features:         %s",
            (feat_opt.use_sibling_linear ?         "true" : "false"));
  TRACE_LOG("Grandchild features:             %s",
            (feat_opt.use_grand ?                  "true" : "false"));
  TRACE_LOG("Grandchild basic features:       %s",
            (feat_opt.use_grand_basic ?            "true" : "false"));
  TRACE_LOG("Grandchild linear features:      %s",
            (feat_opt.use_grand_linear ?           "true" : "false"));

  const char * test_file = test_opt.test_file.c_str();

  ifstream f(test_file);
  if (!f) {
    ERROR_LOG("Failed to load test file %s", test_file);
    return;
  }

  CoNLLReader reader(f);
  CoNLLWriter writer(cout);

  Instance * inst = NULL;

  decoder=build_decoder();
  cerr << get_time() - before << endl;
  before = get_time();

  int head_correct = 0;
  int label_correct = 0;
  int total_rels = 0;

  while ((inst = reader.next())) {
    int len = inst->size();
    if (model_opt.labeled) {
      inst->deprelsidx.resize(len);
      for (int i = 1; i < len; ++ i) {
        inst->deprelsidx[i] = model->deprels.index(inst->deprels[i].c_str());
      }
    }

    extract_features(inst);
    calculate_score(inst, model->param, true);

    decoder->decode(inst);

    if (model_opt.labeled) {
      inst->predicted_deprels.resize(len);
      for (int i = 1; i < len; ++ i) {
        inst->predicted_deprels[i] = model->deprels.at(inst->predicted_deprelsidx[i]);
      }
    }

    writer.write(inst);

    total_rels += inst->num_rels();
    head_correct += inst->num_correct_heads();
    label_correct += inst->num_correct_heads_and_labels();
    delete inst;
  }
  double after = get_time();
  cerr << after - before << endl;

  TRACE_LOG("UAS: %.4lf ( %d / %d )",
      (double)head_correct / total_rels,
      head_correct,
      total_rels);

  if (model_opt.labeled) {
    TRACE_LOG("LAS: %.4lf ( %d / %d )",
        (double)label_correct / total_rels,
        label_correct,
        total_rels);
  }

  //sleep(1000000);
}

// Enumerate all the subtree in the whole tree space (without specifed tree),
// cache the score for each subtree into inst-><type>_scores.
void
Parser::calculate_score(Instance * inst,
                        const Parameters& param,
                        bool use_avg) {
  int len = inst->size();
  int L = model->num_deprels();

  if (feat_opt.use_unlabeled_dependency) {
    for (treeutils::DEPTreeSpaceIterator itx(len); !itx.end(); ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();

      FeatureVector * fv = inst->depu_features[hid][cid];
      inst->depu_scores[hid][cid] = 0.;

      if (!fv) {
        continue;
      }

      inst->depu_scores[hid][cid] = param.dot(fv, use_avg);
    }
  }   //  end if feat_opt.use_unlabeled_dependency

  if (feat_opt.use_labeled_dependency) {
     for (treeutils::DEPTreeSpaceIterator itx(len); !itx.end(); ++ itx) {
       int hid = itx.hid();
       int cid = itx.cid();
       for (int l = 0; l < L; ++ l) {
         FeatureVector * fv = inst->depl_features[hid][cid][l];
         inst->depl_scores[hid][cid][l] = 0.;

         if (!fv) {
           continue;
         }

         inst->depl_scores[hid][cid][l] = param.dot(fv, use_avg);
      }
    }
  }   //  end if feat_opt.use_labeled_dependency

  if (feat_opt.use_unlabeled_sibling) {
    for (treeutils::SIBTreeSpaceIterator itx(len, feat_opt.use_last_sibling);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int sid = itx.sid();

      FeatureVector * fv = inst->sibu_features[hid][cid][sid];
      inst->sibu_scores[hid][cid][sid] = 0.;

      if (!fv) {
        continue;
      }

      inst->sibu_scores[hid][cid][sid] = param.dot(fv, use_avg);
    }
  }   //  end for if feat_opt.use_unlabeled_sibling

  if (feat_opt.use_labeled_sibling) {
    for (treeutils::SIBTreeSpaceIterator itx(len, feat_opt.use_last_sibling);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int sid = itx.sid();

      for (int l = 0; l < L; ++ l) {
        FeatureVector * fv = inst->sibl_features[hid][cid][sid][l];
        inst->sibl_scores[hid][cid][sid][l] = 0.;

        if (!fv) {
          continue;
        }

        inst->sibl_scores[hid][cid][sid][l] = param.dot(fv, use_avg);
      }
    }
  }   //  end for if feat_opt.use_labeled_sibling

  if (feat_opt.use_unlabeled_grand) {
    for (treeutils::GRDTreeSpaceIterator itx(len, feat_opt.use_no_grand);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int gid = itx.gid();

      FeatureVector * fv = inst->grdu_features[hid][cid][gid];
      inst->grdu_scores[hid][cid][gid] = 0.;

      if (!fv) {
        continue;
      }

      inst->grdu_scores[hid][cid][gid] = param.dot(fv, use_avg);
    }
  }   //  end for feat_opt.use_unlabeled_grand

  if (feat_opt.use_labeled_grand) {
    for (treeutils::GRDTreeSpaceIterator itx(len, feat_opt.use_no_grand);
         !itx.end();
         ++ itx) {
      int hid = itx.hid();
      int cid = itx.cid();
      int gid = itx.gid();

      for (int l = 0; l < L; ++ l) {
        FeatureVector * fv = inst->grdl_features[hid][cid][gid][l];
        inst->grdl_scores[hid][cid][gid][l] = 0.;

        if (!fv) {
          continue;
        }

        inst->grdl_scores[hid][cid][gid][l] = param.dot(fv, use_avg);
      }
    }
  }   //  end for use_labeled_grand
}

}   //  end for namespace parser
}   //  end for namespace ltp
