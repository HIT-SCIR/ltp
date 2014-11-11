#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "segmentor/instance.h"
#include "segmentor/extractor.h"
#include "segmentor/options.h"
#include "segmentor/segmentreader.h"
#include "segmentor/segmentwriter.h"
#include "segmentor/customized_segmentor.h"

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
  if (baseline_model) {
    delete baseline_model;
  }
}

bool
CustomizedSegmentor::parse_cfg(ltp::utility::ConfigParser & cfg) {
  Segmentor::parse_cfg(cfg);

  std::string strbuf;

  static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_name = "";
  if (cfg.has_section("train")) {
    int intbuf;

    if(cfg.get("train", "baseline-model-name",strbuf)) {
      static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_name = strbuf;
    } else {
      WARNING_LOG("baseline model name is not configed, [%s] is set as default",
                  static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_name.c_str());
    }
  }

  static_cast<CustomizedTestOptions*>(test_opt)->baseline_model_file = "";
  if (cfg.has_section("test")) {
    if (cfg.get("test", "baseline-model-file", strbuf)) {
      static_cast<CustomizedTestOptions*>(test_opt)->baseline_model_file = strbuf;
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
CustomizedSegmentor::extract_features(Instance* inst, bool create) {
  const int N = Extractor::num_templates();
  const int L = baseline_model->num_labels();
  const int len = inst->size();

  baseline_uni_features.resize(len, L);
  baseline_uni_features = 0;

  std::vector< utils::StringVec > cache;
  cache.resize(N);
  std::vector< int > cache_again;

  if (0 == inst->lexicon_match_state.size()) {
    inst->lexicon_match_state.resize(len, 0);

    for (int i = 0; i < len; ++ i) {
      std::string word; word.reserve(32);
      for (int j = i; j<i+5 && j < len; ++ j) {
        word = word + inst->forms[j];

        // it's not a lexicon word
        if (!model->internal_lexicon.get(word.c_str())
            && !model->external_lexicon.get(word.c_str())
            && !baseline_model->external_lexicon.get(word.c_str())
            && !baseline_model->internal_lexicon.get(word.c_str())
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

  // cache lexicon features.
  for (int pos = 0; pos < len; ++ pos) {
    for (int n = 0; n < N; ++ n) {
      cache[n].clear();
    }
    cache_again.clear();

    Extractor::extract1o(inst, pos, cache);

    if (baseline_model->labels.size() > 0) {
      for (int tid = 0; tid < cache.size(); ++ tid) {
        for (int itx = 0; itx < cache[tid].size(); ++ itx) {
          if (create) {
            baseline_model->space.retrieve(tid, cache[tid][itx], false);
          }

          int idx = baseline_model->space.index(tid, cache[tid][itx]);

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

        baseline_uni_features[pos][l] = new math::FeatureVector;
        baseline_uni_features[pos][l]->n = num_feat;
        baseline_uni_features[pos][l]->val = 0;
        baseline_uni_features[pos][l]->loff = 0;
        baseline_uni_features[pos][l]->idx = idx;

        for (l = 1; l < L; ++ l) {
          baseline_uni_features[pos][l] = new math::FeatureVector;
          baseline_uni_features[pos][l]->n = num_feat;
          baseline_uni_features[pos][l]->idx = idx;
          baseline_uni_features[pos][l]->val = 0;
          baseline_uni_features[pos][l]->loff = l;
        }
      }
    }
  }

  Segmentor::extract_features(inst, create);
}

void
CustomizedSegmentor::calculate_scores(Instance * inst, bool use_avg) {
  //bool use_avg = 0;
  int len = inst->size();
  int L = model->num_labels();

  inst->uni_scores.resize(len, L);  inst->uni_scores = NEG_INF;
  inst->bi_scores.resize(L, L);     inst->bi_scores = NEG_INF;

  for (int i = 0; i < len; ++ i) {
    for (int l = 0; l < L; ++ l) {
      math::FeatureVector * fv = baseline_uni_features[i][l];
      if (!fv) {
        continue;
      }

      if(!use_avg) {
        inst->uni_scores[i][l] = baseline_model->param.dot(fv, false);
      } else {
        inst->uni_scores[i][l] = baseline_model->param.dot_flush_time(fv,
            baseline_model->end_time,
            model->end_time);
      }

      fv = uni_features[i][l];
      if (!fv) {
        continue;
      }
      inst->uni_scores[i][l] += model->param.dot(uni_features[i][l], use_avg);
      //std::cout<<"uni_scores["<<i<<"]["<<l<<"]="<<inst->uni_scores[i][l]<<std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int idx = baseline_model->space.index(pl, l);

      if(!use_avg) {
        inst->bi_scores[pl][l] = baseline_model->param.dot(idx, false);
      } else {
        inst->bi_scores[pl][l] = baseline_model->param.dot_flush_time(idx, 
            baseline_model->end_time,
            model->end_time);
      }

      idx = model->space.index(pl, l);
      inst->bi_scores[pl][l] += model->param.dot(idx, use_avg);
      //std::cout<<"bi_scores["<<pl<<"]["<<l<<"]="<<inst->bi_scores[pl][l]<<std::endl;
    }
  }
}

void
CustomizedSegmentor::collect_correct_and_predicted_features(Instance * inst) {
  collect_features(uni_features, model, inst, inst->tagsidx,
                   correct_features);
  collect_features(uni_features, model, inst, inst->predicted_tagsidx,
                   predicted_features);
  updated_features.zero();
  updated_features.add(correct_features, 1.);
  updated_features.add(predicted_features, -1.);

  collect_features(baseline_uni_features, baseline_model, inst, inst->tagsidx,
                   baseline_correct_features);
  collect_features(baseline_uni_features, baseline_model, inst, inst->predicted_tagsidx,
                   baseline_predicted_features);
  baseline_updated_features.zero();
  baseline_updated_features.add(baseline_correct_features, 1.);
  baseline_updated_features.add(baseline_predicted_features, -1.);
}

bool
CustomizedSegmentor::train_setup() {
  const char * baseline_model_file =
    static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_name.c_str();

  std::ifstream mfs(baseline_model_file, std::ifstream::binary);
  if(!mfs) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  baseline_model = new Model;
  if (!baseline_model->load(mfs)) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  if (!baseline_model->full) {
    ERROR_LOG("Baseline model must be full");
    return false;
  }

  return true;
}

int
CustomizedSegmentor::get_timestamp(void) {
  return timestamp + baseline_model->end_time;
}

void
CustomizedSegmentor::train_passive_aggressive(int nr_errors) {
  double error = nr_errors;
  double score = (model->param.dot(updated_features, false)
      + baseline_model->param.dot(baseline_updated_features, false));
  double norm  = updated_features.L2();

  double step = 0.;
  if (norm < EPS) {
    step = 0;
  } else {
    step = (error - score) / norm;
  }

  model->param.add(updated_features, timestamp + baseline_model->end_time, step);
}

void
CustomizedSegmentor::train_averaged_perceptron() {
  model->param.add(updated_features, timestamp + baseline_model->end_time, 1.);
}

void
CustomizedSegmentor::evaluate(double &p, double &r, double &f) {
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

bool
CustomizedSegmentor::test_setup() {
  const char * baseline_model_file =
    static_cast<CustomizedTestOptions *>(test_opt)->baseline_model_file.c_str();

  std::ifstream mfs(baseline_model_file, std::ifstream::binary);
  if(!mfs) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  baseline_model = new Model;
  if (!baseline_model->load(mfs)) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  return true;
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
