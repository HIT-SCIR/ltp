#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "segmentor/customized_segmentor.h"
#include "segmentor/instance.h"
#include "segmentor/extractor.h"
#include "segmentor/options.h"
#include "segmentor/segmentreader.h"
#include "segmentor/segmentwriter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>    //  std::sort
#include <functional>   //  std::greater

#if _WIN32
#include <Windows.h>
#define sleep Sleep
#endif  //  end for _WIN32

namespace ltp {
namespace segmentor {

namespace utils = ltp::utility;

CustomizedSegmentor::CustomizedSegmentor()
  : baseline_model(0) {
}

CustomizedSegmentor::CustomizedSegmentor(utils::ConfigParser& cfg)
  : baseline_model(0) {
  train_opt = new CustomizedTrainOptions;
  test_opt = new CustomizedTestOptions;
  dump_opt = new CustomizedDumpOptions;
  parse_cfg(cfg);
}

CustomizedSegmentor::~CustomizedSegmentor() {
  if (personal_model) {
    delete personal_model;
  }
}

bool
CustomizedSegmentor::parse_cfg(ltp::utility::ConfigParser & cfg) {
  Segmentor::parse_cfg(cfg);

  std::string strbuf;

  train_opt->baseline_model_name = "";
  if (cfg.has_section("train")) {
    int intbuf;

    if(cfg.get("train", "baseline-model-name",strbuf)) {
      train_opt->baseline_model_name = strbuf;
    } else {
      WARNING_LOG("baseline model name is not configed, [%s] is set as default",
                  train_opt->baseline_model_name.c_str());
    }
  }

  test_opt.baseline_model_name = "";
  if (cfg.has_section("test")) {
    if (cfg.get("test", "baseline-model-file", strbuf)) {
      test_opt->baseline_model_name= strbuf;
    } else {
      ERROR_LOG("personal-model-file is not configed.");
      return false;
    }
  }
  return true;
}

void
CustomizedSegmentor::build_configuration(void) {
  // First, inherit the tag set from the baseline model.
  for (int i = 0; i < baseline_model->labels.size(); ++i) {
    const char * key = baseline_model->labels.at(i);
    model->labels.push(key);
  }

  Segmentor::build_configuration();
}

void
CustomizedSegmentor::extract_features(Instance * inst, bool create) {
  const int N = Extractor::num_templates();
  const int L = model->num_labels();

  vector< StringVec > cache;
  vector< int > cache_again;

  cache.resize(N);
  int len = inst->size();

  // allocate the uni_features
  inst->uni_features.resize(len, L);  inst->uni_features = 0;
  inst->personal_uni_features.resize(len, L);  inst->personal_uni_features = 0;
  inst->uni_scores.resize(len, L);  inst->uni_scores = NEG_INF;
  inst->bi_scores.resize(L, L);     inst->bi_scores = NEG_INF;

  // cache lexicon features.
  if (0 == inst->lexicon_match_state.size()) {
    inst->lexicon_match_state.resize(len, 0);

    for (int i = 0; i < len; ++ i) {
      std::string word; word.reserve(32);
      for (int j = i; j<i+5 && j < len; ++ j) {
        word = word + inst->forms[j];

        // it's not a lexicon word
        if (!model->internal_lexicon.get(word.c_str())
            && !model->external_lexicon.get(word.c_str())
            && !personal_model->external_lexicon.get(word.c_str())
            && !personal_model->internal_lexicon.get(word.c_str())
            ) {
          continue;
        }

        int l = j+1-i;

        if (l > (inst->lexicon_match_state[i] & 0x0F)) {
          inst->lexicon_match_state[i] &= 0xfff0;
          inst->lexicon_match_state[i] |= l;
        }

        if (l > ((inst->lexicon_match_state[j]>>4) & 0x0F)) {
          inst->lexicon_match_state[j] &= 0xff0f;
          inst->lexicon_match_state[j] |= (l<<4);
        }

        for (int k = i+1; k < j; ++k) {
          if (l>((inst->lexicon_match_state[k]>>8) & 0x0F)) {
            inst->lexicon_match_state[k] &= 0xf0ff;
            inst->lexicon_match_state[k] |= (l<<8);
          }
        }
      }
    }
  }

  for (int pos = 0; pos < len; ++ pos) {
    for (int n = 0; n < N; ++ n) {
      cache[n].clear();
    }
    cache_again.clear();

    Extractor::extract1o(inst, pos, cache);

    if (personal_model->labels.size() > 0) {
      for (int tid = 0; tid < cache.size(); ++ tid) {
        for (int itx = 0; itx < cache[tid].size(); ++ itx) {
          if (create) {
            personal_model->space.retrieve(tid, cache[tid][itx], true);
          }

          int idx = personal_model->space.index(tid, cache[tid][itx]);

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

        inst->personal_uni_features[pos][l] = new FeatureVector;
        inst->personal_uni_features[pos][l]->n = num_feat;
        inst->personal_uni_features[pos][l]->val = 0;
        inst->personal_uni_features[pos][l]->loff = 0;
        inst->personal_uni_features[pos][l]->idx = idx;

        for (l = 1; l < L; ++ l) {
          inst->personal_uni_features[pos][l] = new FeatureVector;
          inst->personal_uni_features[pos][l]->n = num_feat;
          inst->personal_uni_features[pos][l]->idx = idx;
          inst->personal_uni_features[pos][l]->val = 0;
          inst->personal_uni_features[pos][l]->loff = l;
        }
      }

      cache_again.clear();
    }

    for (int tid = 0; tid < cache.size(); ++tid) {
      for (int itx = 0; itx < cache[tid].size(); ++itx) {
        int idx = model->space.index(tid, cache[tid][itx]);
        if (idx >= 0) {
          cache_again.push_back(idx);
        }
      }
    }

    int num_feat = cache_again.size();

    if (num_feat > 0) {
      int l = 0;
      int * idx = new int [num_feat];
      for (int j = 0; j < num_feat; ++j) {
        idx[j] = cache_again[j];
      }

      inst->uni_features[pos][l] = new FeatureVector;
      inst->uni_features[pos][l]->n = num_feat;
      inst->uni_features[pos][l]->val = 0;
      inst->uni_features[pos][l]->loff = 0;
      inst->uni_features[pos][l]->idx = idx;

      for (l = 1; l < L; ++l) {
      inst->uni_features[pos][l] = new FeatureVector;
      inst->uni_features[pos][l]->n = num_feat;
      inst->uni_features[pos][l]->val = 0;
      inst->uni_features[pos][l]->loff = l;
      inst->uni_features[pos][l]->idx = idx;
      }
    }
  }
}

void
CustomizedSegmentor::build_feature_space(void) {
  // build feature space, it a wrapper for
  // featurespace.build_feature_space
  int L = personal_model->num_labels();
  personal_model->space.set_num_labels(L);

  for (int i = 0; i < train_dat.size(); ++ i) {
    extract_features(train_dat[i], true);
    if ((i + 1) % train_opt.display_interval == 0) {
      TRACE_LOG("[%d] instances is extracted.", (i+1));
    }
  }
}

void
CustomizedSegmentor::calculate_scores(Instance * inst, bool use_avg) {
  int len = inst->size();
  int L = model->num_labels();
  for (int i = 0; i < len; ++ i) {
    for (int l = 0; l < L; ++ l) {
      FeatureVector * fv = inst->uni_features[i][l];
      if (!fv) {
        continue;
      }

      if(!use_avg) {
        inst->uni_scores[i][l] = model->param.dot(inst->uni_features[i][l], false);
      }
      else {
        inst->uni_scores[i][l] = model->param.dot_flush_time(inst->uni_features[i][l], model->end_time, personal_model->end_time);
      }

      fv = inst->personal_uni_features[i][l];
      if (!fv) {
        continue;
      }
      inst->uni_scores[i][l] += personal_model->param.dot(inst->personal_uni_features[i][l], use_avg);
      //std::cout<<"uni_scores["<<i<<"]["<<l<<"]="<<inst->uni_scores[i][l]<<std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int idx = model->space.index(pl, l);
      if(!use_avg) {
        inst->bi_scores[pl][l] = model->param.dot(idx, false);
      }
      else {
        inst->bi_scores[pl][l] = model->param.dot_flush_time(idx, model->end_time, personal_model->end_time);
      }
      idx = personal_model->space.index(pl, l);
      inst->bi_scores[pl][l] += personal_model->param.dot(idx, use_avg);
      //std::cout<<"bi_scores["<<pl<<"]["<<l<<"]="<<inst->bi_scores[pl][l]<<std::endl;
    }
  }
}

void
CustomizedSegmentor::collect_features(Instance * inst,
    const std::vector<int> & tagsidx,
    math::SparseVec & vec,
    math::SparseVec & personal_vec) {
  int len = inst->size();

  vec.zero();
  personal_vec.zero();
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

    const FeatureVector * personal_fv = inst->personal_uni_features[i][l];

    if(!personal_fv) {
      continue;
    }
    personal_vec.add(personal_fv->idx, personal_fv->val, personal_fv->n, personal_fv->loff, 1.);

    if ( i > 0) {
      int prev_lid = tagsidx[i-1];
      int idx = personal_model->space.index(prev_lid, l);
      personal_vec.add(idx, 1.);
    }
  }
}

// Perform model truncation on the model, according to these two conditions
//  (1) Erase the group of parameters that it all the parameter in this group
//      is equals to zero.
//  (2) (optional) Erase the group of parameters the total updated time is
//      lower than the pre-defined threshold.

Model *
CustomizedSegmentor::erase_rare_features(const int * feature_updated_times) {
  Model * new_model = new Model;
  // copy the label indexable map to the new model
  for (int i = 0; i < personal_model->labels.size(); ++ i) {
    const char * key = personal_model->labels.at(i);
    new_model->labels.push(key);
  }

  TRACE_LOG("building labels map is done");

  int L = new_model->num_labels();
  new_model->space.set_num_labels(L);

  // Iterate over the feature space
  for (FeatureSpaceIterator itx = personal_model->space.begin();
      itx != personal_model->space.end();
      ++ itx) {
    const char * key = itx.key();
    int tid = itx.tid();
    int id  = personal_model->space.index(tid, key);

    // Enumerate each feature with the same feature prefix, check if it's zero.
    bool flag = false;
    for (int l = 0; l < L; ++ l) {
      double p = personal_model->param.dot(id + l);
      if (p != 0.) {
        flag = true;
      }
    }

    if (!flag) {
      continue;
    }

    // Check if this feature's updated time.
    int idx = personal_model->space.retrieve(tid, key, false);
    if (feature_updated_times
        && (feature_updated_times[idx] < train_opt.rare_feature_threshold)) {
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

    int old_id = personal_model->space.index(tid, key);
    int new_id = new_model->space.index(tid, key);

    for (int l = 0; l < L; ++ l) {
      // pay attention to this place, use average should be set true
      // some dirty code
      new_model->param._W[new_id + l]      = personal_model->param._W[old_id + l];
      new_model->param._W_sum[new_id + l]  = personal_model->param._W_sum[old_id + l];
      new_model->param._W_time[new_id + l] = personal_model->param._W_time[old_id + l];
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int old_id = personal_model->space.index(pl, l);
      int new_id = new_model->space.index(pl, l);

      new_model->param._W[new_id]      = personal_model->param._W[old_id];
      new_model->param._W_sum[new_id]  = personal_model->param._W_sum[old_id];
      new_model->param._W_time[new_id] = personal_model->param._W_time[old_id];
    }
  }
  TRACE_LOG("Building new model is done");

  for (SmartMap<bool>::const_iterator itx = personal_model->internal_lexicon.begin();
      itx != personal_model->internal_lexicon.end();
      ++ itx) {
    new_model->internal_lexicon.set(itx.key(), true);
  }
  new_model->end_time = personal_model->end_time;

  return new_model;
}

void
Personal_Segmentor::train(void) {
  const char * train_file = train_opt.train_file.c_str();
  const char * model_file = train_opt.model_name.c_str();
  ifstream mfs(model_file, std::ifstream::binary);

  if(!mfs) {
    ERROR_LOG("Failed to load basic model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load basic model");
    return;
  }

  // read in training instance
  personal_model = new Model;
  if (!read_instance(train_file)) {
    ERROR_LOG("Training file not exist.");
    return;
  }
  TRACE_LOG("Read in [%d] personal instances.", train_dat.size());

  // build tag dictionary, map string tag to index
  TRACE_LOG("Start build configuration");
  build_configuration();
  TRACE_LOG("Build configuration is done.");
  TRACE_LOG("Number of labels: [%d]", personal_model->labels.size());

  // build feature space from the training instance
  TRACE_LOG("Start building feature space.");
  build_feature_space();
  TRACE_LOG("Building feature space is done.");
  TRACE_LOG("Number of features: [%d]", personal_model->space.num_features());

  personal_model->param.realloc(personal_model->space.dim());
  TRACE_LOG("Allocate [%d] dimensition parameter.", personal_model->space.dim());

  int nr_feature_groups = personal_model->space.num_feature_groups();
  int * feature_group_updated_time = NULL;

  // If the rare feature threshold is used, allocate memory for the
  // feature group updated time.
  if (train_opt.rare_feature_threshold > 0) {
    feature_group_updated_time = new int[nr_feature_groups];
    for (int i = 0; i < nr_feature_groups; ++ i) {
      feature_group_updated_time[i] = 0;
    }
  }

  TRACE_LOG("Allocate [%d] update counters.", nr_feature_groups);

  SegmentWriter writer(cout);

  if (train_opt.algorithm == "mira") {
    // use mira to train model
    // it's still not implemented.
  } else {
    // use pa or average perceptron algorithm
    rulebase::RuleBase base(personal_model->labels);
    decoder = new Decoder(personal_model->num_labels(), base);
    TRACE_LOG("Allocated plain decoder");

    int best_iteration = -1;
    double best_p = -1.;
    double best_r = -1.;
    double best_f = -1.;

    for (int iter = 0; iter < train_opt.max_iter; ++ iter) {
      TRACE_LOG("Training iteraition [%d]", (iter + 1));

      for (int i = 0; i < train_dat.size(); ++ i) {
        // extract_features(train_dat[i]);

        Instance * inst = train_dat[i];
        calculate_scores(inst, false);
        decoder->decode(inst);

        if (inst->features.dim() == 0) {
          collect_features(inst, inst->tagsidx, inst->features, inst->personal_features);
        }
        collect_features(inst, inst->predicted_tagsidx, inst->predicted_features, inst->personal_predicted_features);

        if (train_opt.algorithm == "pa") {
          SparseVec update_features;
          SparseVec personal_update_features;

          update_features.zero();
          update_features.add(inst->features, 1.);
          update_features.add(inst->predicted_features, -1.);

          personal_update_features.zero();
          personal_update_features.add(inst->personal_features, 1.);
          personal_update_features.add(inst->personal_predicted_features, -1.);

          if (feature_group_updated_time) {
            Segmentor::increase_group_updated_time(update_features,
                                        feature_group_updated_time);
          }

          double error = inst->num_errors();
          double score = personal_model->param.dot(personal_update_features, false);
          score += model->param.dot(update_features, false);
          double norm  = personal_update_features.L2();

          double step = 0.;
          if (norm < EPS) {
            step = 0;
          } else {
            step = (error - score) / norm;
          }

          personal_model->param.add(personal_update_features,
                           iter * train_dat.size() + i + 1 + model->end_time,
                           step);

        } else if (train_opt.algorithm == "ap") {
          SparseVec update_features;
          update_features.zero();
          update_features.add(train_dat[i]->features, 1.);
          update_features.add(train_dat[i]->predicted_features, -1.);

          if (feature_group_updated_time) {
            Segmentor::increase_group_updated_time(update_features,
                                        feature_group_updated_time);
          }

          model->param.add(update_features,
                           iter * train_dat.size() + i + 1,
                           1.);
        }

        if ((i+1) % train_opt.display_interval == 0) {
          TRACE_LOG("[%d] instances is trained.", i+1);
        }
      }
      personal_model->param.flush( train_dat.size() * (iter + 1) + model->end_time);
      personal_model->end_time = train_dat.size() * (iter + 1) + model->end_time;

      Model * new_model = NULL;
      new_model = erase_rare_features(feature_group_updated_time);

      double p, r, f;
      swap(new_model, personal_model);
      evaluate(p,r,f);

      if (f > best_f) {
        best_p = p;
        best_r = r;
        best_f = f;
        best_iteration = iter;
      }

      std::string saved_model_file = (train_opt.personal_model_name
                                      + "."
                                      + strutils::to_str(iter)
                                      + ".model");
      std::ofstream ofs(saved_model_file.c_str(), std::ofstream::binary);

      swap(new_model, personal_model);
      new_model->save(ofs);
      delete new_model;
      TRACE_LOG("Model for iteration [%d] is saved to [%s]",
                iter + 1,
                saved_model_file.c_str());
    }

    if (feature_group_updated_time) {
      delete [] feature_group_updated_time;
    }

    TRACE_LOG("Best result (iteratin = %d) P = %lf | R = %lf | F = %lf",
              best_iteration,
              best_p,
              best_r,
              best_f);
  }
}

void
CustomizedSegmentor::evaluate(double &p, double &r, double &f) {
  const char * holdout_file = train_opt.holdout_file.c_str();

  ifstream ifs(holdout_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  SegmentReader reader(ifs, true);
  Instance * inst = NULL;

  int num_recalled_words = 0;
  int num_predicted_words = 0;
  int num_gold_words = 0;

  int beg_tag0 = personal_model->labels.index( __b__ );
  int beg_tag1 = personal_model->labels.index( __s__ );

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    for (int i = 0; i < len; ++ i) {
      inst->tagsidx[i] = personal_model->labels.index(inst->tags[i]);
    }

    extract_features(inst);
    calculate_scores(inst, true);

    decoder->decode(inst);

    if (inst->words.size() == 0) {
      Segmentor::build_words(inst, inst->tagsidx, inst->words, beg_tag0, beg_tag1);
    }
    Segmentor::build_words(inst,
                inst->predicted_tagsidx,
                inst->predicted_words,
                beg_tag0,
                beg_tag1);

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

void
CustomizedSegmentor::test(void) {
  // load model
  const char * model_file = test_opt.model_file.c_str();
  ifstream mfs(model_file, std::ifstream::binary);

  if (!mfs) {
    ERROR_LOG("Failed to load basic model");
    return;
  }

  model = new Model;
  if (!model->load(mfs)) {
    ERROR_LOG("Failed to load basic model");
    return;
  }
  const char * personal_model_file = test_opt.personal_model_file.c_str();
  ifstream p_mfs(personal_model_file, std::ifstream::binary);

  if(!p_mfs) {
    ERROR_LOG("Failed to load personal model");
    return;
  }

  personal_model = new Model;
  if (!personal_model->load(p_mfs)) {
    ERROR_LOG("Failed to load personal model");
    return;
  }

  // load exteranl lexicon
  const char * lexicon_file =test_opt.lexicon_file.c_str();

  if (NULL != lexicon_file) {
    ifstream lfs(lexicon_file);

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

  const char * test_file = test_opt.test_file.c_str();

  ifstream ifs(test_file);

  if (!ifs) {
    ERROR_LOG("Failed to open holdout file.");
    return;
  }

  rulebase::RuleBase base(model->labels);
  Decoder * decoder = new Decoder(model->num_labels(), base);
  SegmentReader reader(ifs, true);
  SegmentWriter writer(cout);
  Instance * inst = NULL;

  int beg_tag0 = model->labels.index( __b__ );
  int beg_tag1 = model->labels.index( __s__ );
  int num_recalled_words = 0;
  int num_predicted_words = 0;
  int num_gold_words = 0;

  double before = get_time();

  while ((inst = reader.next())) {
    int len = inst->size();
    inst->tagsidx.resize(len);
    for (int i = 0; i < len; ++i) {
      inst->tagsidx[i] = model->labels.index(inst->tags[i]);
    }

    extract_features(inst);
    calculate_scores(inst, true);
    decoder->decode(inst);

    if (inst->words.size()==0) {
      Segmentor::build_words(inst, inst->tagsidx, inst->words, beg_tag0, beg_tag1);
    }

    Segmentor::build_words(inst,
                inst->predicted_tagsidx,
                inst->predicted_words,
                beg_tag0,
                beg_tag1);
    num_recalled_words += inst->num_recalled_words();
    num_predicted_words += inst->num_predicted_words();
    num_gold_words += inst->num_gold_words();

    writer.write(inst);
    delete inst;
  }

  double p = (double)num_recalled_words / num_predicted_words;
  double r = (double)num_recalled_words / num_gold_words;
  double f = 2 * p * r / (p+r);
  double after = get_time();
  TRACE_LOG("Eclipse time %lf", after - before);
  TRACE_LOG("P: %lf",p);
  TRACE_LOG("R: %lf",r);
  TRACE_LOG("F: %lf",f);
  return;
}

void
CustomizedSegmentor::dump() {
  WARNING_LOG("not implemented.");
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
