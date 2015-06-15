#include "utils/time.hpp"
#include "utils/logging.hpp"
#include "segmentor/customized_segmentor_frontend.h"
#include "segmentor/instance.h"
#include "segmentor/extractor.h"
#include "segmentor/options.h"
#include "segmentor/io.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>    //  std::sort
#include <functional>   //  std::greater

namespace ltp {
namespace segmentor {

using ltp::math::SparseVec;

CustomizedSegmentorFrontend::CustomizedSegmentorFrontend(
  const std::string& reference_file,
  const std::string& holdout_file,
  const std::string& model_name,
  const std::string& baseline_model_file,
  const std::string& algorithm,
  const size_t& max_iter,
  const size_t& rare_feature_threshold)
  : bs_model(0), bs_model_file(baseline_model_file),
  SegmentorFrontend(reference_file, holdout_file, model_name,
  algorithm, max_iter, rare_feature_threshold, true) {
  good = load_baseline_model();
  timestamp = bs_model->param.last();
}

CustomizedSegmentorFrontend::CustomizedSegmentorFrontend(
  const std::string& input_file,
  const std::string& model_file,
  const std::string& baseline_model_file,
  bool evaluate)
  : bs_model(0), bs_model_file(baseline_model_file),
  SegmentorFrontend(input_file, model_file, evaluate) {
  good = load_baseline_model();
  timestamp = bs_model->param.last();
}

CustomizedSegmentorFrontend::~CustomizedSegmentorFrontend() {
  if (bs_model) { delete bs_model; bs_model = 0; }
}

bool CustomizedSegmentorFrontend::load_baseline_model() {
  std::ifstream mfs(bs_model_file.c_str(), std::ifstream::binary);
  if(!mfs) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  bs_model = new Model;
  if (!bs_model->load(Segmentor::model_header.c_str(), mfs)) {
    ERROR_LOG("Failed to load baseline model");
    return false;
  }

  if (bs_model->param.is_wrapper()) {
    ERROR_LOG("Baseline model should be fully dumped version.");
    WARNING_LOG("Model trained by LTP version lower than 3.2.0 or be");
    WARNING_LOG("- configed by 'enable-incremental-training = 1' is not fully dumped.");
    WARNING_LOG("Please retrain your model with LTP above 3.2.0 and ");
    WARNING_LOG("- 'enable-incremental-training = 1' ");
    return false;
  }

  INFO_LOG("report: baseline model, number of labels = %d", bs_model->num_labels());
  INFO_LOG("report: baseline model, number of features = %d", bs_model->space.num_features());
  INFO_LOG("report: baseline model, number of dimension = %d", bs_model->space.dim());
  return true;
}

void CustomizedSegmentorFrontend::setup_lexicons() {
  lexicons.push_back(&(model->internal_lexicon));
  lexicons.push_back(&(model->external_lexicon));
  lexicons.push_back(&(bs_model->internal_lexicon));
}

void CustomizedSegmentorFrontend::extract_features(const Instance& inst, bool create) {
  //extract features from model
  Segmentor::extract_features(inst, model, &ctx, create);
  //extract features from baseline model
  Segmentor::extract_features(inst, bs_model, &bs_ctx, false);
}

void CustomizedSegmentorFrontend::clear_context(void) {
  ctx.clear();
  bs_ctx.clear();
}

void CustomizedSegmentorFrontend::build_configuration(void) {
  // First, inherit the tag set from the baseline model.
  for (size_t i = 0; i < bs_model->labels.size(); ++i) {
    const char* key = bs_model->labels.at(i);
    model->labels.push(key);
  }

  SegmentorFrontend::build_configuration();
}

void CustomizedSegmentorFrontend::calculate_scores(const Instance& inst, bool avg) {
  //bool use_avg = 0;
  Segmentor::calculate_scores(inst, *bs_model, *model, bs_ctx, ctx, avg, &scm);
}

void CustomizedSegmentorFrontend::collect_features(const Instance& inst) {
  Frontend::collect_features(model, ctx.uni_features,
      inst.tagsidx, ctx.correct_features);
  Frontend::collect_features(model, ctx.uni_features,
      inst.predict_tagsidx, ctx.predict_features);

  Frontend::collect_features(bs_model, bs_ctx.uni_features,
      inst.tagsidx, bs_ctx.correct_features);
  Frontend::collect_features(bs_model, bs_ctx.uni_features,
      inst.predict_tagsidx, bs_ctx.predict_features);
}

void CustomizedSegmentorFrontend::update(const Instance& inst, SparseVec& updated_features) {
  updated_features.zero();
  updated_features.add(ctx.correct_features, 1.);
  updated_features.add(ctx.predict_features, -1.);

  if (train_opt.algorithm == "pa") {
    SparseVec bs_updated_features;
    bs_updated_features.add(bs_ctx.correct_features, 1.);
    bs_updated_features.add(bs_ctx.predict_features, -1.);

    double error = (double)InstanceUtils::num_errors(inst.tagsidx, inst.predict_tagsidx);
    double score = (model->param.dot(updated_features, false) +
       bs_model->param.dot(bs_updated_features, false));

    Frontend::learn_passive_aggressive(updated_features, get_timestamp(), error, score, model);
  } else {
    Frontend::learn_averaged_perceptron(updated_features, get_timestamp(), model);
  }
}

}     //  end for namespace segmentor
}     //  end for namespace ltp
