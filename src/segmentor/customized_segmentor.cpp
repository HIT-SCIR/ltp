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
  : baseline_model(0),
  baseline_decode_context(0){
}

CustomizedSegmentor::CustomizedSegmentor(utils::ConfigParser& cfg)
  : baseline_model(0),
  baseline_decode_context(0){
  train_opt = new CustomizedTrainOptions;
  test_opt = new CustomizedTestOptions;
  dump_opt = new CustomizedDumpOptions;
  parse_cfg(cfg);
}

CustomizedSegmentor::~CustomizedSegmentor() {
  if (baseline_model) {
    delete baseline_model;
  }

  if (baseline_decode_context) {
    delete baseline_decode_context;
  }
}

bool
CustomizedSegmentor::parse_cfg(ltp::utility::ConfigParser & cfg) {

  std::string strbuf;

  if (cfg.has_section("train")) {
    int intbuf;

    if(cfg.get("train", "baseline-model-file",strbuf)) {
      static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_file = strbuf;
    } else {
      ERROR_LOG("baseline model file is not configed.");
      return false;
    }

    if(cfg.get("train", "customized-model-name",strbuf)) {
      cfg.set("train", "model-name",strbuf);
    }
  }

  static_cast<CustomizedTestOptions*>(test_opt)->baseline_model_file = "";
  if (cfg.has_section("test")) {
    if (cfg.get("test", "baseline-model-file", strbuf)) {
      static_cast<CustomizedTestOptions*>(test_opt)->baseline_model_file = strbuf;
    } else {
      ERROR_LOG("baseline-model-file is not configed.");
      return false;
    }

    if (cfg.get("test", "customized-model-file", strbuf)) {
      cfg.set("test", "model-file",strbuf);
    } else {
      ERROR_LOG("customized-model-file is not configed.");
      return false;
    }
  }

  if (!Segmentor::parse_cfg(cfg)) {
    return false;
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
CustomizedSegmentor::build_lexicon_match_state(Instance* inst) {
  // cache lexicon features.
  int len = inst->size();

  if (inst->lexicon_match_state.size()) {
    return;
  }

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


void
CustomizedSegmentor::extract_features(const Instance* inst, bool create) {
  Segmentor::extract_features(inst, model, decode_context, create);
  Segmentor::extract_features(inst, baseline_model, baseline_decode_context, false);
}

void
CustomizedSegmentor::calculate_scores(const Instance* inst, bool use_avg) {
  calculate_scores(inst, decode_context, baseline_decode_context, use_avg, score_matrix);
}
void
CustomizedSegmentor::calculate_scores(const Instance * inst, const DecodeContext* ctx, const DecodeContext * base_ctx, bool use_avg, ScoreMatrix* scm) {
  //bool use_avg = 0;
  int len = inst->size();
  int L = model->num_labels();

  scm->uni_scores.resize(len, L);
  scm->uni_scores = NEG_INF;
  scm->bi_scores.resize(L, L);
  scm->bi_scores = NEG_INF;

  for (int i = 0; i < len; ++ i) {
    for (int l = 0; l < L; ++ l) {
      math::FeatureVector * fv = base_ctx->uni_features[i][l];
      if (!fv) {
        continue;
      }

      if(!use_avg) {
        scm->uni_scores[i][l] = baseline_model->param.dot(fv, false);
      } else {
        scm->uni_scores[i][l] = baseline_model->param.dot_flush_time(fv,
            baseline_model->end_time,
            model->end_time);
      }

      fv = ctx->uni_features[i][l];
      if (!fv) {
        continue;
      }
      scm->uni_scores[i][l] += model->param.dot(fv,
          use_avg);
      //std::cout<<"uni_scores["<<i<<"]["<<l<<"]="<<inst->uni_scores[i][l]<<std::endl;
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int idx = baseline_model->space.index(pl, l);

      if(!use_avg) {
        scm->bi_scores[pl][l] = baseline_model->param.dot(idx, false);
      } else {
        scm->bi_scores[pl][l] = baseline_model->param.dot_flush_time(idx, 
            baseline_model->end_time,
            model->end_time);
      }

      idx = model->space.index(pl, l);
      scm->bi_scores[pl][l] += model->param.dot(idx, use_avg);
      //std::cout<<"bi_scores["<<pl<<"]["<<l<<"]="<<inst->bi_scores[pl][l]<<std::endl;
    }
  }
}

void
CustomizedSegmentor::collect_correct_and_predicted_features(Instance * inst) {
  DecodeContext* ctx = decode_context;
  collect_features(ctx->uni_features, model, inst->tagsidx,
                   ctx->correct_features);
  collect_features(ctx->uni_features, model, inst->predicted_tagsidx,
                   ctx->predicted_features);
  ctx->updated_features.zero();
  ctx->updated_features.add(ctx->correct_features, 1.);
  ctx->updated_features.add(ctx->predicted_features, -1.);

  ctx = baseline_decode_context;
  collect_features(ctx->uni_features, baseline_model, inst->tagsidx,
                   ctx->correct_features);
  collect_features(ctx->uni_features, baseline_model, inst->predicted_tagsidx,
                   ctx->predicted_features);
  ctx->updated_features.zero();
  ctx->updated_features.add(ctx->correct_features, 1.);
  ctx->updated_features.add(ctx->predicted_features, -1.);
}

bool
CustomizedSegmentor::train_setup() {
  const char * baseline_model_file =
    static_cast<CustomizedTrainOptions *>(train_opt)->baseline_model_file.c_str();

  std::ifstream mfs(baseline_model_file, std::ifstream::binary);
  if(!mfs) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  baseline_decode_context = new DecodeContext;
  baseline_model = new Model;
  if (!baseline_model->load(mfs)) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  if (!baseline_model->full) {
    ERROR_LOG("Baseline model should be fully dumped version.");
    WARNING_LOG("Model trained by LTP version lower than 3.2.0 or be");
    WARNING_LOG("- configed by 'enable-incremental-training = 1' is not fully dumped.");
    WARNING_LOG("Please retrain your model with LTP above 3.2.0 and ");
    WARNING_LOG("- 'enable-incremental-training = 1' ");
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
  double score = (model->param.dot(decode_context->updated_features, false)
      + baseline_model->param.dot(baseline_decode_context->updated_features, false));
  double norm  = decode_context->updated_features.L2();

  double step = 0.;
  if (norm < EPS) {
    step = 0;
  } else {
    step = (error - score) / norm;
  }

  model->param.add(decode_context->updated_features, timestamp + baseline_model->end_time, step);
}

void
CustomizedSegmentor::train_averaged_perceptron() {
  model->param.add(decode_context->updated_features, timestamp + baseline_model->end_time, 1.);
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
  baseline_decode_context = new DecodeContext;

  return true;
}

void
CustomizedSegmentor::cleanup_decode_context() {
  decode_context->clear();
  baseline_decode_context->clear();
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
