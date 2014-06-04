#include "ner.h"

#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "ner/instance.h"
#include "ner/extractor.h"
#include "ner/options.h"
#include "ner/nerio.h"

#include <iostream>
#include <sstream>
#include <fstream>

#if _WIN32
#include <Windows.h>
#define sleep Sleep
#endif	//	end for _WIN32

namespace ltp {
namespace ner {

NER::NER() 
  : model(0),
    decoder(0),
    __TRAIN__(false),
    __TEST__(false),
    __DUMP__(false) {
}

NER::NER(ltp::utility::ConfigParser & cfg)
  : model(0),
    decoder(0),
    __TRAIN__(false),
    __TEST__(false),
    __DUMP__(false) {
  parse_cfg(cfg);
}

NER::~NER() {
  if (model) {
    delete model;
  }

  if (decoder) {
    delete decoder;
  }
}

void
NER::run(void) {
  if (__TRAIN__) {
    train();
  }

  if (__TEST__) {
    test();
  }

  if (__DUMP__) {
    dump();
  }

  for (int i = 0; i < train_dat.size(); ++ i) {
    if (train_dat[i]) {
      delete train_dat[i];
    }
  }
}

bool
NER::parse_cfg(ltp::utility::ConfigParser & cfg) {
  std::string strbuf;
  int         intbuf;


  train_opt.train_file        = "";
  train_opt.holdout_file      = "";
  train_opt.algorithm         = "pa";
  train_opt.model_name        = "";
  train_opt.max_iter          = 10;
  train_opt.display_interval  = 5000;

  if (cfg.has_section("train")) {
    TRACE_LOG("Training mode specified.");
    __TRAIN__ = true;

    if (cfg.get("train", "train-file", strbuf)) {
      train_opt.train_file = strbuf;
    } else {
      ERROR_LOG("train-file config item is not found.");
      return false;
    }

    if (cfg.get("train", "holdout-file", strbuf)) {
      train_opt.holdout_file = strbuf;
    } else {
      ERROR_LOG("holdout-file config item is not found.");
      return false;
    }

    if (cfg.get("train", "algorithm", strbuf)) {
      train_opt.algorithm = strbuf;
    } else {
      WARNING_LOG("algorithm is not configed, [PA] is set as default");
    }

    train_opt.model_name = train_opt.train_file + "." + train_opt.algorithm;
    if (cfg.get("train", "model-name", strbuf)) {
      train_opt.model_name = strbuf;
    } else {
      WARNING_LOG("model name is not configed, [%s] is set as default",
          train_opt.model_name.c_str());
    }

    if (cfg.get_integer("train", "max-iter", intbuf)) {
      train_opt.max_iter = intbuf;
    } else {
      WARNING_LOG("max-iter is not configed, [10] is set as default.");
    }
  }

  test_opt.test_file = "";
  test_opt.model_file = "";
  test_opt.lexicon_file = "";

  if (cfg.has_section("test")) {
    __TEST__ = true;

    if (cfg.get("test", "test-file", strbuf)) {
      test_opt.test_file = strbuf;
    } else {
      ERROR_LOG("test-file config item is not set.");
      return false;
    }

    if (cfg.get("test", "model-file", strbuf)) {
      test_opt.model_file = strbuf;
    } else {
      ERROR_LOG("model-file is not configed. ");
      return false;
    }

    if (cfg.get("test", "lexicon-file", strbuf)) {
      test_opt.lexicon_file = strbuf;
    }
  }

  dump_opt.model_file = "";
  if (cfg.has_section("dump")) {
    __DUMP__ = true;

    if (cfg.get("dump", "model-file", strbuf)) {
      dump_opt.model_file = strbuf;
    } else {
      ERROR_LOG("model-file is not configed.");
      return false;
    }
  }

  return true;
}

bool
NER::read_instance(const char * train_file) {
  std::ifstream ifs(train_file);

  if (!ifs) {
    return false;
  }

  NERReader reader(ifs, true);
  train_dat.clear();

  Instance * inst = NULL;

  while ((inst = reader.next())) {
    train_dat.push_back(inst);
  }

  return true;
}

void NER::build_configuration(void) {
  // tag set is some kind of hard coded into the source

  std::stringstream S;
  for (int i = 0; i < __num_pos_types__; ++ i) {
    for (int j = 0; j < __num_ne_types__; ++ j) {
      S.str(std::string());
      S << __pos_types__[i] << "-" << __ne_types__[j];
      model->labels.push(S.str());
    }
  }
  model->labels.push("O");

  for (int i = 0; i < train_dat.size(); ++ i) {
    Instance * inst = train_dat[i];
    int len = inst->size();

    inst->tagsidx.resize(len);
    for (int j = 0; j < len; ++ j) {
      // build labels dictionary
      inst->tagsidx[j] = model->labels.index( inst->tags[j] );
    }
  }
}

void NER::extract_features(Instance * inst, bool create) {
  const int N = Extractor::num_templates();
  const int L = model->num_labels();

  vector< StringVec > cache;
  vector< int > cache_again;

  cache.resize(N);
  int len = inst->size();

  // allocate the uni_features
  inst->uni_features.resize(len, L);  inst->uni_features = 0;
  inst->uni_scores.resize(len, L);    inst->uni_scores = NEG_INF;
  inst->bi_scores.resize(L, L);       inst->bi_scores = NEG_INF;

  for (int pos = 0; pos < len; ++ pos) {
    for (int n = 0; n < N; ++ n) {
      cache[n].clear();
    }
    cache_again.clear();

    Extractor::extract1o(inst, pos, cache);

    for (int tid = 0; tid < cache.size(); ++ tid) {
      for (int itx = 0; itx < cache[tid].size(); ++ itx) {
        if (create) {
          model->space.retrieve(tid, cache[tid][itx], true);
        }

        int idx = model->space.index(tid, cache[tid][itx]);

        if (idx >= 0) {
          cache_again.push_back(idx);
        }
      }
    }

    int num_feat = cache_again.size();

    if (num_feat > 0) {
      int l = 0;
      int * idx = new int[num_feat];
      for (int j = 0; j < num_feat; ++ j) {
        idx[j] = cache_again[j];
      }

      inst->uni_features[pos][l] = new FeatureVector;
      inst->uni_features[pos][l]->n = num_feat;
      inst->uni_features[pos][l]->val = 0;
      inst->uni_features[pos][l]->loff = 0;
      inst->uni_features[pos][l]->idx = idx;

      for (l = 1; l < L; ++ l) {
        inst->uni_features[pos][l] = new FeatureVector;
        inst->uni_features[pos][l]->n = num_feat;
        inst->uni_features[pos][l]->idx = idx;
        inst->uni_features[pos][l]->val = 0;
        inst->uni_features[pos][l]->loff = l;
      }
    }
  }
}

void NER::build_feature_space(void) {
  // build feature space, it a wrapper for
  // featurespace.build_feature_space
  Extractor::num_templates();

  int L = model->num_labels();
  model->space.set_num_labels(L);

  for (int i = 0; i < train_dat.size(); ++ i) {
    extract_features(train_dat[i], true);
    if ((i + 1) % train_opt.display_interval == 0) {
      TRACE_LOG("[%d] instances is extracted.", (i+1));
    }
  }
}

void
NER::build_entities(Instance * inst,
                    const std::vector<int> & tagsidx,
                    std::vector<std::string> & entities,
                    std::vector<std::string> & entities_tags,
                    int beg_tag0,
                    int beg_tag1,
                    int beg_tag2) {
  entities.clear();
  entities_tags.clear();

  int len = inst->size();

  // should check the tagsidx size
  std::string entity = inst->raw_forms[0];

  int tag = inst->tagsidx[0];
  int tag_prefix = -1;
  int tag_suffix = tag % __num_ne_types__;

  std::string entity_tag = (tag == 12 ? "O" : __ne_types__[tag_suffix]);
  for (int i = 1; i < len; ++ i) {
    tag = tagsidx[i];

    tag_prefix = tag / __num_ne_types__;
    tag_suffix = (tag % __num_ne_types__);

    if (tag_prefix == beg_tag0 || tag_prefix == beg_tag1 || tag_prefix == beg_tag2) {
      entities.push_back(entity);
      entities_tags.push_back(entity_tag);

      entity = inst->raw_forms[i];
      entity_tag = (tag == 12 ? "O" : __ne_types__[tag_suffix]);
    } else {
      entity += inst->raw_forms[i];
    }
  }
  entities.push_back(entity);
  entities_tags.push_back(entity_tag);
}

void
NER::calculate_scores(Instance * inst, bool use_avg) {
  int len = inst->size();
  int L = model->num_labels();
  for (int i = 0; i < len; ++ i) {
    for (int l = 0; l < L; ++ l) {
      FeatureVector * fv = inst->uni_features[i][l];
      if (!fv) {
        continue;
      }

      inst->uni_scores[i][l] = model->param.dot(inst->uni_features[i][l], use_avg);
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int idx = model->space.index(pl, l);
      inst->bi_scores[pl][l] = model->param.dot(idx, use_avg);
    }
  }
}

void
NER::collect_features(Instance * inst,
                      const std::vector<int> & tagsidx,
                      math::SparseVec & vec) {
  int len = inst->size();

  vec.zero();
  for (int i = 0; i < len; ++ i) {
    int l = tagsidx[i];
    const FeatureVector * fv = inst->uni_features[i][l];

    if (!fv) {
      continue;
    }

    vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);

    if (i > 0) {
      int prev_lid = tagsidx[i-1];
      int idx = model->space.index(prev_lid, l);
      vec.add(idx, 1.);
    }
  }
}

Model *
NER::truncate(void) {
  Model * new_model = new Model;
  // copy the label indexable map to the new model
  for (int i = 0; i < model->labels.size(); ++ i) {
    const char * key = model->labels.at(i);
    new_model->labels.push(key);
  }

  TRACE_LOG("building labels map is done");

  int L = new_model->num_labels();
  new_model->space.set_num_labels(L);

  // iterate over the feature space and see if the parameter value equals to zero
  for (FeatureSpaceIterator itx = model->space.begin();
      itx != model->space.end();
      ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);

    bool flag = false;
    for (int l = 0; l < L; ++ l) {
      double p = model->param.dot(id + l);
      if (p != 0.) {
        flag = true;
      }
    }

    if (!flag) {
      continue;
    }

    new_model->space.retrieve(tid, key, true);
  }
  TRACE_LOG("Scanning old features space, building new feature space is done");

  new_model->param.realloc(new_model->space.dim());
  TRACE_LOG("Parameter dimension of new model is [%d]", new_model->space.dim());

  for (FeatureSpaceIterator itx = new_model->space.begin();
      itx != new_model->space.end();
      ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();

    int old_id = model->space.index(tid, key);
    int new_id = new_model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      // pay attention to this place, use average should be set true
      // some dirty code
      new_model->param._W[new_id + l]     = model->param._W[old_id + l];
      new_model->param._W_sum[new_id + l]   = model->param._W_sum[old_id + l];
      new_model->param._W_time[new_id + l]  = model->param._W_time[old_id + l];
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int old_id = model->space.index(pl, l);
      int new_id = new_model->space.index(pl, l);

      new_model->param._W[new_id]     = model->param._W[old_id];
      new_model->param._W_sum[new_id]   = model->param._W_sum[old_id];
      new_model->param._W_time[new_id]  = model->param._W_time[old_id];
    }
  }
  TRACE_LOG("Building new model is done");

  for (SmartMap<int>::const_iterator itx = model->cluster_lexicon.begin();
      itx != model->cluster_lexicon.end();
      ++ itx) {
    new_model->cluster_lexicon.set(itx.key(), (*itx.value()));
  }

  return new_model;
}

void
NER::train(void) {
  const char * train_file = train_opt.train_file.c_str();

  // read in training instance
  if (!read_instance(train_file)) {
    ERROR_LOG("Training file doesn't exist");
  }

  TRACE_LOG("Read in [%d] instances.", train_dat.size());

  model = new Model;
  // build tag dictionary, map string tag to index
  TRACE_LOG("Start build configuration");
  build_configuration();
  TRACE_LOG("Build configuration is done.");
  TRACE_LOG("Number of labels: [%d]", model->labels.size());

  // build feature space from the training instance
  TRACE_LOG("Start building feature space.");
  build_feature_space();
  TRACE_LOG("Building feature space is done.");
  TRACE_LOG("Number of features: [%d]", model->space.num_features());

  model->param.realloc(model->space.dim());
  TRACE_LOG("Allocate [%d] dimensition parameter.", model->space.dim());

  NERWriter writer(std::cout);

  if (train_opt.algorithm == "mira") {
    // use mira algorithm
    /*kbest_decoder = new KBestDecoder(L);

    for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
      for (int i = 0; i < train_dat.size(); ++ i) {
        extract_features(train_dat[i]);
        calculate_scores(train_dat[i]);

        KBestDecoder::KBestDecodeResult result;
        kbest_decoder->decode(train_dat[i], result);
      }
    }*/
  } else {
    // use pa or average perceptron algorithm
    rulebase::RuleBase base(model->labels);
    decoder = new Decoder(model->num_labels(), base);
    TRACE_LOG("Allocated plain decoder");

    for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
      TRACE_LOG("Training iteraition [%d]", (iter + 1));
      for (int i = 0; i < train_dat.size(); ++ i) {
        // extract_features(train_dat[i]);

        Instance * inst = train_dat[i];
        calculate_scores(inst, false);
        decoder->decode(inst);

        if (inst->features.dim() == 0) {
          collect_features(inst, inst->tagsidx, inst->features);
        }
        collect_features(inst, inst->predicted_tagsidx, inst->predicted_features);


        if (train_opt.algorithm == "pa") {
          SparseVec update_features;
          update_features.zero();
          update_features.add(train_dat[i]->features, 1.);
          update_features.add(train_dat[i]->predicted_features, -1.);

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
          update_features.zero();
          update_features.add(train_dat[i]->features, 1.);
          update_features.add(train_dat[i]->predicted_features, -1.);

          model->param.add(update_features,
                           iter * train_dat.size() + i + 1,
                           1.);
        }

        if ((i+1) % train_opt.display_interval == 0) {
          TRACE_LOG("[%d] instances is trained.", i+1);
        }
      }
      model->param.flush( train_dat.size() * (iter + 1) );

      Model * new_model = truncate();
      swap(model, new_model);
      evaluate();

      std::string saved_model_file = (train_opt.model_name
                                      + "."
                                      + strutils::to_str(iter)
                                      + ".model");
      std::ofstream ofs(saved_model_file.c_str(), std::ofstream::binary);

      swap(model, new_model);
      new_model->save(ofs);
      delete new_model;

      TRACE_LOG("Model for iteration [%d] is saved to [%s]",
          iter + 1,
          saved_model_file.c_str());
    }
  }
}

void NER::evaluate(void) {
  const char * holdout_file = train_opt.holdout_file.c_str();

  ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  NERReader reader(ifs, true);
  NERWriter writer(std::cout);
  Instance * inst = NULL;

  // some dirty hard code and trick
  int beg_tag0 = (model->labels.index( "B-Nh" ) / __num_ne_types__);
  int beg_tag1 = (model->labels.index( "S-Nh" ) / __num_ne_types__);
  int beg_tag2 = (model->labels.index( "O" ) / __num_ne_types__);


  int num_recalled_entities = 0;
  int num_predicted_entities = 0;
  int num_gold_entities = 0;

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
    }

    extract_features(inst);
    calculate_scores(inst, true);
    decoder->decode(inst);

    // writer.debug(inst);
    if (inst->entities.size() == 0) {
      build_entities(inst,
          inst->tagsidx,
          inst->entities,
          inst->entities_tags,
          beg_tag0,
          beg_tag1,
          beg_tag2);
    }

    build_entities(inst,
        inst->predicted_tagsidx,
        inst->predicted_entities,
        inst->predicted_entities_tags,
        beg_tag0,
        beg_tag1,
        beg_tag2);

    num_recalled_entities += inst->num_recalled_entites();
    num_predicted_entities += inst->num_predicted_entities();
    num_gold_entities += inst->num_gold_entities();

    delete inst;
  }

  double p = (double)num_recalled_entities / num_predicted_entities;
  double r = (double)num_recalled_entities / num_gold_entities;
  double f = 2 * p * r / (p + r);

  TRACE_LOG("P: %lf ( %d / %d )", p, num_recalled_entities, num_predicted_entities);
  TRACE_LOG("R: %lf ( %d / %d )", r, num_recalled_entities, num_gold_entities);
  TRACE_LOG("F: %lf" , f);
  return;
}

void NER::test(void) {
  // load model
  const char * model_file = test_opt.model_file.c_str();
  ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  TRACE_LOG("Number of labels         [%d]", model->num_labels());
  TRACE_LOG("Number of features       [%d]", model->space.num_features());
  TRACE_LOG("Number of dimension      [%d]", model->space.dim());

  const char * test_file = test_opt.test_file.c_str();

  ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  rulebase::RuleBase base(model->labels);
  decoder = new Decoder(model->num_labels(), base);
  NERReader reader(ifs);
  NERWriter writer(cout);
  Instance * inst = NULL;

  // int beg_tag0 = model->labels.index( __b__ );
  // int beg_tag1 = model->labels.index( __s__ );

  double before = get_time();

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);

    extract_features(inst);
    calculate_scores(inst, true);
    decoder->decode(inst);

    writer.write(inst);
    delete inst;
  }

  double after = get_time();
  TRACE_LOG("Eclipse time %lf", after - before);

  sleep(1000000);
  return;
}

void NER::dump() {
  // load model
  const char * model_file = dump_opt.model_file.c_str();
  ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  int L = model->num_labels();
  TRACE_LOG("Number of labels         [%d]", model->num_labels());
  TRACE_LOG("Number of features       [%d]", model->space.num_features());
  TRACE_LOG("Number of dimension      [%d]", model->space.dim());

  for (FeatureSpaceIterator itx = model->space.begin();
       itx != model->space.end();
       ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      std::cout << key << " ( " << id + l << " ) "
                << " --> "
                << model->param.dot(id + l)
                << std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int id = model->space.index(pl, l);
      std::cout << pl << " --> " << l << " " << model->param.dot(id) << std::endl;
    }
  }
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
