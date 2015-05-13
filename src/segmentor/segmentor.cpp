#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "utils/strvec.hpp"
#include "segmentor/segmentor.h"
#include "segmentor/instance.h"
#include "segmentor/extractor.h"
#include "segmentor/options.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>    //  std::sort
#include <functional>   //  std::greater

namespace ltp {
namespace segmentor {

using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;
using utility::StringVec;
using utility::SmartMap;
using math::FeatureVector;
using math::SparseVec;

Segmentor::Segmentor(): model(0) {}
Segmentor::~Segmentor() { if (model) { delete model; model = 0; } }

void Segmentor::extract_features(const Instance& inst,
    Model* mdl,
    ViterbiFeatureContext* ctx,
    bool create) const {
  size_t N = Extractor::num_templates();
  size_t T = mdl->num_labels();

  std::vector< StringVec > cache;
  std::vector< int > cache_again;

  cache.resize(N);
  size_t L = inst.size();

  if (ctx) {
    // allocate the uni_features
    ctx->uni_features.resize(L, T);
    ctx->uni_features = 0;
  }

  for (size_t pos = 0; pos < L; ++ pos) {
    for (size_t n = 0; n < N; ++ n) { cache[n].clear(); }
    cache_again.clear();

    Extractor::extract1o(inst, pos, cache);
    for (size_t tid = 0; tid < cache.size(); ++ tid) {
      for (size_t itx = 0; itx < cache[tid].size(); ++ itx) {
        if (create) { mdl->space.retrieve(tid, cache[tid][itx], true); }

        int idx = mdl->space.index(tid, cache[tid][itx]);
        if (idx >= 0) { cache_again.push_back(idx); }
      }
    }

    size_t num_feat = cache_again.size();
    if (num_feat > 0 && ctx) {
      size_t t = 0;
      int * idx = new int[num_feat];
      for (size_t j = 0; j < num_feat; ++ j) {
        idx[j] = cache_again[j];
      }

      ctx->uni_features[pos][t] = new FeatureVector;
      ctx->uni_features[pos][t]->n = num_feat;
      ctx->uni_features[pos][t]->val = 0;
      ctx->uni_features[pos][t]->loff = 0;
      ctx->uni_features[pos][t]->idx = idx;

      for (t = 1; t < T; ++ t) {
        ctx->uni_features[pos][t] = new FeatureVector;
        ctx->uni_features[pos][t]->n = num_feat;
        ctx->uni_features[pos][t]->idx = idx;
        ctx->uni_features[pos][t]->val = 0;
        ctx->uni_features[pos][t]->loff = t;
      }
    }
  }
}

void Segmentor::build_lexicon_match_state(
    const std::vector<const Model::lexicon_t*>& lexicons,
    Instance* inst) const {
  // cache lexicon features.
  size_t len = inst->size();
  if (inst->lexicon_match_state.size()) { return; }
  inst->lexicon_match_state.resize(len, 0);

  // perform the maximum forward match algorithm
  for (size_t i = 0; i < len; ++ i) {
    std::string word; word.reserve(32);
    for (size_t j = i; j<i+5 && j < len; ++ j) {
      word = word + inst->forms[j];

      bool found = false;
      for (size_t k = 0; k < lexicons.size(); ++ k) {
        if (lexicons[k]->get(word.c_str())) {
          found = true; break;
        }
      }

      if (!found) { continue; }
      int l = j+1-i;

      if (l > (inst->lexicon_match_state[i] & 0x0F)) {
        inst->lexicon_match_state[i] &= 0xfff0;
        inst->lexicon_match_state[i] |= l;
      }

      if (l > ((inst->lexicon_match_state[j]>>4) & 0x0F)) {
        inst->lexicon_match_state[j] &= 0xff0f;
        inst->lexicon_match_state[j] |= (l<<4);
      }

      for (size_t k = i+1; k < j; ++k) {
        if (l>((inst->lexicon_match_state[k]>>8) & 0x0F)) {
          inst->lexicon_match_state[k] &= 0xf0ff;
          inst->lexicon_match_state[k] |= (l<<8);
        }
      }
    }
  }
}

void Segmentor::calculate_scores(const Instance& inst,
    const Model& mdl,
    const ViterbiFeatureContext& ctx,
    bool avg,
    ViterbiScoreMatrix* scm) {
  size_t L = inst.size();
  size_t T = mdl.num_labels();

  scm->resize(L, T, -1e20);

  for (size_t i = 0; i < L; ++ i) {
    for (size_t t = 0; t < T; ++ t) {
      FeatureVector * fv = ctx.uni_features[i][t];
      if (!fv) {
        continue;
      }
      scm->set_emit(i, t, mdl.param.dot(ctx.uni_features[i][t], avg));
    }
  }

  for (size_t pt = 0; pt < T; ++ pt) {
    for (size_t t = 0; t < T; ++ t) {
      int idx = mdl.space.index(pt, t);
      scm->set_tran(pt, t, mdl.param.dot(idx, avg));
    }
  }
}

void Segmentor::build_words(const Instance& inst,
    const std::vector<int>& tagsidx,
    std::vector<std::string>& words) {
  words.clear();
  int len = inst.size();

  // should check the tagsidx size
  std::string word = inst.raw_forms[0];
  for (int i = 1; i < len; ++ i) {
    int tag = tagsidx[i];
    if (tag == 0 || tag == 3) { // b, s
      words.push_back(word);
      word = inst.raw_forms[i];
    } else {
      word += inst.raw_forms[i];
    }
  }

  words.push_back(word);
}


#if 0
void
Segmentor::run(void) {
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
Segmentor::parse_cfg(utils::ConfigParser & cfg) {
  std::string strbuf;

  train_opt->train_file = "";
  train_opt->holdout_file = "";
  train_opt->algorithm = "pa";
  train_opt->model_name = "";
  train_opt->max_iter = 10;
  train_opt->display_interval = 5000;
  train_opt->rare_feature_threshold = 0;
  train_opt->enable_incremental_training = 0;

  if (cfg.has_section("train")) {
    int intbuf;

    TRACE_LOG("Training mode specified.");
    __TRAIN__ = true;

    if (cfg.get("train", "train-file", strbuf)) {
      train_opt->train_file = strbuf;
    } else {
      ERROR_LOG("train-file config item is not found.");
      return false;
    }

    if (cfg.get("train", "holdout-file", strbuf)) {
      train_opt->holdout_file = strbuf;
    } else {
      ERROR_LOG("holdout-file config item is not found.");
      return false;
    }

    if (cfg.get("train", "algorithm", strbuf)) {
      train_opt->algorithm = strbuf;
    } else {
      WARNING_LOG("algorithm is not configed, [PA] is set as default");
    }

    if (cfg.get("train", "rare-feature-threshold", strbuf)) {
      train_opt->rare_feature_threshold = atoi(strbuf.c_str());
    } else {
      WARNING_LOG("rare feature threshold is not configed, 10 is set as default");
    }

    train_opt->model_name = train_opt->train_file + "." + train_opt->algorithm;
    if (cfg.get("train", "model-name", strbuf)) {
      train_opt->model_name = strbuf;
    } else {
      WARNING_LOG("model name is not configed, [%s] is set as default",
                  train_opt->model_name.c_str());
    }

    if (cfg.get_integer("train", "max-iter", intbuf)) {
      train_opt->max_iter = intbuf;
    } else {
      WARNING_LOG("max-iter is not configed, [10] is set as default.");
    }

    if (cfg.get_integer("train", "enable-incremental-training", intbuf)) {
      train_opt->enable_incremental_training = (intbuf == 1);
    }
  }

  test_opt->test_file    = "";
  test_opt->model_file   = "";
  test_opt->lexicon_file = "";

  if (cfg.has_section("test")) {
    __TEST__ = true;

    if (cfg.get("test", "test-file", strbuf)) {
      test_opt->test_file = strbuf;
    } else {
      ERROR_LOG("test-file config item is not set.");
      return false;
    }

    if (cfg.get("test", "model-file", strbuf)) {
      test_opt->model_file = strbuf;
    } else {
      ERROR_LOG("model-file is not configed.");
      return false;
    }

    if (cfg.get("test", "lexicon-file", strbuf)) {
      test_opt->lexicon_file = strbuf;
    }
  }

  dump_opt->model_file = "";
  if (cfg.has_section("dump")) {
    __DUMP__ = true;

    if (cfg.get("dump", "model-file", strbuf)) {
      dump_opt->model_file = strbuf;
    } else {
      ERROR_LOG("model-file is not configed.");
      return false;
    }
  }

  return true;
}

void
Segmentor::extract_features(const Instance* inst, bool create) {
  extract_features(inst, model, decode_context, create);
}

void
Segmentor::calculate_scores(const Instance* inst, bool use_avg) {
  calculate_scores(inst, model, decode_context, use_avg, score_matrix);
}

void
Segmentor::collect_features(const math::Mat< math::FeatureVector* >& features,
    const Model* mdl, const std::vector<int> & tagsidx, math::SparseVec & vec) {

  int len = tagsidx.size();
  vec.zero();
  for (int i = 0; i < len; ++ i) {
    int l = tagsidx[i];
    const math::FeatureVector * fv = features[i][l];

    if (!fv) {
      continue;
    }

    vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);

    if (i > 0) {
      int prev_lid = tagsidx[i-1];
      int idx = mdl->space.index(prev_lid, l);
      vec.add(idx, 1.);
    }
  }
}

// Perform model truncation on the model, according to these two conditions
//  (1) Erase the group of parameters that it all the parameter in this group
//      is equals to zero.
//  (2) (optional) Erase the group of parameters the total updated time is
//      lower than the pre-defined threshold.
Model *
Segmentor::erase_rare_features(const int * feature_updated_times) {
  Model * new_model = new Model;
  // copy the label indexable map to the new model
  for (int i = 0; i < model->labels.size(); ++ i) {
    const char * key = model->labels.at(i);
    new_model->labels.push(key);
  }

  TRACE_LOG("building labels map is done");

  int L = new_model->num_labels();
  new_model->space.set_num_labels(L);

  // Iterate over the feature space
  for (FeatureSpaceIterator itx = model->space.begin();
      itx != model->space.end();
      ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id  = model->space.index(tid, key);

    // Enumerate each feature with the same feature prefix, check if it's zero.
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

    // Check if this feature's updated time.
    int idx = model->space.retrieve(tid, key, false);
    if (feature_updated_times
        && (feature_updated_times[idx] < train_opt->rare_feature_threshold)) {
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
      new_model->param._W[new_id + l]      = model->param._W[old_id + l];
      new_model->param._W_sum[new_id + l]  = model->param._W_sum[old_id + l];
      new_model->param._W_time[new_id + l] = model->param._W_time[old_id + l];
    }
  }

  // copy parameters from the old model to the new model
  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int old_id = model->space.index(pl, l);
      int new_id = new_model->space.index(pl, l);

      new_model->param._W[new_id]      = model->param._W[old_id];
      new_model->param._W_sum[new_id]  = model->param._W_sum[old_id];
      new_model->param._W_time[new_id] = model->param._W_time[old_id];
    }
  }
  TRACE_LOG("Building new model is done");

  // copy lexicon from the old model to the new model
  for (utils::SmartMap<bool>::const_iterator itx = model->internal_lexicon.begin();
      itx != model->internal_lexicon.end();
      ++ itx) {
    new_model->internal_lexicon.set(itx.key(), true);
  }
  new_model->end_time = model->end_time;
  new_model->full = model->full;

  return new_model;
}

void
Segmentor::collect_correct_and_predicted_features(Instance* inst) {
  collect_features(decode_context->uni_features, model, inst->tagsidx,
      decode_context->correct_features);
  collect_features(decode_context->uni_features, model, inst->predicted_tagsidx,
      decode_context->predicted_features);

  decode_context->updated_features.zero();
  decode_context->updated_features.add(decode_context->correct_features, 1.);
  decode_context->updated_features.add(decode_context->predicted_features, -1.);
}

void
Segmentor::train_passive_aggressive(int nr_errors) {
  //double error = train_dat[i]->num_errors();
  double error = nr_errors;
  double score = model->param.dot(decode_context->updated_features, false);
  double norm  = decode_context->updated_features.L2();

  double step = 0.;
  if (norm < EPS) {
    step = 0;
  } else {
    step = (error - score) / norm;
  }

  model->param.add(decode_context->updated_features, timestamp, step);
}

void
Segmentor::train_averaged_perceptron() {
  model->param.add(decode_context->updated_features, timestamp, 1.);
}

bool
Segmentor::train_setup() {
  return true;
};

int
Segmentor::get_timestamp(void) {
  return timestamp;
}

void
Segmentor::set_timestamp(int ts) {
  timestamp = ts;
}

void
Segmentor::cleanup_decode_context() {
  decode_context->clear();
}

void
Segmentor::evaluate(double &p, double &r, double &f) {
  const char * holdout_file = train_opt->holdout_file.c_str();

  std::ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  SegmentReader reader(ifs, true);
  Instance * inst = NULL;

  int num_recalled_words = 0;
  int num_predicted_words = 0;
  int num_gold_words = 0;

  int beg_tag0 = model->labels.index( __b__ );
  int beg_tag1 = model->labels.index( __s__ );

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
    }

    build_lexicon_match_state(inst);
    extract_features(inst);
    calculate_scores(inst, true);
    decoder->decode(inst, score_matrix);
    cleanup_decode_context();

    build_words(inst, inst->tagsidx, inst->words, beg_tag0, beg_tag1);
    build_words(inst, inst->predicted_tagsidx, inst->predicted_words, beg_tag0, beg_tag1);

    num_recalled_words += inst->num_recalled_words();
    num_predicted_words += inst->num_predicted_words();
    num_gold_words += inst->num_gold_words();

    delete inst;
  }

  p = (double)num_recalled_words / num_predicted_words;
  r = (double)num_recalled_words / num_gold_words;
  f = 2 * p * r / (p + r);

  TRACE_LOG("P: %lf ( %d / %d )", p, num_recalled_words, num_predicted_words);
  TRACE_LOG("R: %lf ( %d / %d )", r, num_recalled_words, num_gold_words);
  TRACE_LOG("F: %lf" , f);
  return;
}

bool
Segmentor::test_setup() {
  return true;
}

void
Segmentor::test(void) {
  if (!test_setup()) {
    return;
  }

  // load model
  const char * model_file = test_opt->model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load model");
    return;
  }

  TRACE_LOG("Number of labels     [%d]", model->num_labels());
  TRACE_LOG("Number of features   [%d]", model->space.num_features());
  TRACE_LOG("Number of dimension  [%d]", model->space.dim());

  // load exteranl lexicon
  const char * lexicon_file =test_opt->lexicon_file.c_str();

  if (NULL != lexicon_file) {
    std::ifstream lfs(lexicon_file);

    if (lfs) {
      std::string buffer;
      while (std::getline(lfs, buffer)) {
        buffer = strutils::chomp(buffer);
        if (buffer.size() == 0) {
          continue;
        }
        model->external_lexicon.set(buffer.c_str(), true);
      }
    }
  }

  const char * test_file = test_opt->test_file.c_str();
  std::ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  rulebase::RuleBase base(model->labels);
  decoder = new Decoder(model->num_labels(), base);
  score_matrix = new ScoreMatrix;
  decode_context = new DecodeContext;
  SegmentReader reader(ifs);
  SegmentWriter writer(std::cout);
  Instance * inst = NULL;

  int beg_tag0 = model->labels.index( __b__ );
  int beg_tag1 = model->labels.index( __s__ );

  double before = utils::get_time();

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);

    build_lexicon_match_state(inst);
    extract_features(inst);
    calculate_scores(inst, true);
    decoder->decode(inst, score_matrix);
    cleanup_decode_context();

    build_words(inst, inst->predicted_tagsidx, inst->predicted_words,
        beg_tag0, beg_tag1);

    writer.write(inst);
    delete inst;
  }

  double after = utils::get_time();
  TRACE_LOG("Eclipse time %lf", after - before);
  return;
}


void Segmentor::dump() {
  // load model
  const char * model_file = dump_opt->model_file.c_str();
  std::ifstream mfs(model_file, std::ifstream::binary);

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
  TRACE_LOG("Number of labels     [%d]", model->num_labels());
  TRACE_LOG("Number of features   [%d]", model->space.num_features());
  TRACE_LOG("Number of dimension  [%d]", model->space.dim());

  for (FeatureSpaceIterator itx = model->space.begin();
       itx != model->space.end();
       ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id = model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      std::cout << key
                << " ( " << id + l << " ) "
                << " --> "
                << model->param.dot(id + l)
                << std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int id = model->space.index(pl, l);
      std::cout << pl << " --> " << l
                << " " << model->param.dot(id)
                << std::endl;
    }
  }
}
#endif

}     //  end for namespace segmentor
}     //  end for namespace ltp
