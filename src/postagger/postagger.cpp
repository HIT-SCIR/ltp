#include "postagger/postagger.h"
#include "postagger/extractor.h"
#include "postagger/options.h"
#include "postagger/settings.h"

#include <iostream>
#include <fstream>
#include <iomanip>

namespace ltp {
namespace postagger {

using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;
using utility::StringVec;
using math::FeatureVector;

const std::string Postagger::model_header = "otpos";

Postagger::Postagger(): model(0) {}
Postagger::~Postagger() { if (model) { delete model; model = 0; } }

void Postagger::extract_features(const Instance& inst,
    ViterbiFeatureContext* ctx,
    bool create) const {
  size_t N = Extractor::num_templates();
  size_t T = model->num_labels();
  size_t L = inst.size();

  std::vector< StringVec > cache(N);
  std::vector< int > cache_again;

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
        if (create) { model->space.retrieve(tid, cache[tid][itx], true); }

        int idx = model->space.index(tid, cache[tid][itx]);
        if (idx >= 0) { cache_again.push_back(idx); }
      }
    }

    size_t num_feat = cache_again.size();
    if (num_feat > 0 && ctx) {
      size_t t = 0;
      int* idx = new int[num_feat];
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

void Postagger::calculate_scores(const Instance& inst,
    const ViterbiFeatureContext& ctx, bool avg,
    ViterbiScoreMatrix* scm) const {
  size_t L = inst.size();
  size_t T = model->num_labels();

  scm->resize(L, T, -1e20);

  for (size_t i = 0; i < L; ++ i) {
    for (size_t t = 0; t < T; ++ t) {
      FeatureVector * fv = ctx.uni_features[i][t];
      if (!fv) { continue; }
      scm->set_emit(i, t, model->param.dot(ctx.uni_features[i][t], avg));
    }
  }

  for (size_t pt = 0; pt < T; ++ pt) {
    for (size_t t = 0; t < T; ++ t) {
      int idx = model->space.index(pt, t);
      scm->set_tran(pt, t, model->param.dot(idx, avg));
    }
  }
}

void Postagger::build_labels(const Instance& inst,
    std::vector<std::string>& tags) const {
  size_t len = inst.size();
  if (inst.predict_tagsidx.size() != len) {
    return;
  }

  tags.resize(len);
  for (size_t i = 0; i < len; ++ i) {
    tags[i] = model->labels.at(inst.predict_tagsidx[i]);
  }
}

}     //  end for namespace postagger
}     //  end for namespace ltp
