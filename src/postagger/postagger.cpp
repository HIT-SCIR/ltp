#include "postagger/postagger.h"
#include "postagger/extractor.h"
#include "postagger/options.h"
#include "postagger/settings.h"

#include <iostream>
#include <fstream>
#include <iomanip>

#if _WIN32
#include <Windows.h>
#define sleep Sleep
#endif  //  end for _WIN32

namespace ltp {
namespace postagger {

using utility::StringVec;
using math::FeatureVector;

Postagger::Postagger(): model(0) {}
Postagger::~Postagger() {
  if (model) { delete model; model = 0; }
}

void Postagger::extract_features(const Instance* inst,
    DecodeContext* ctx,
    bool create) const {
  const int N = Extractor::num_templates();
  const int L = model->num_labels();

  std::vector< StringVec > cache;
  std::vector< int > cache_again;

  cache.resize(N);
  int len = inst->size();

  if (ctx) {
    // allocate the uni_features
    ctx->uni_features.resize(len, L);
    ctx->uni_features = 0;
  }

  for (int pos = 0; pos < len; ++ pos) {
    for (int n = 0; n < N; ++ n) { cache[n].clear(); }
    cache_again.clear();

    Extractor::extract1o(inst, pos, cache);
    for (int tid = 0; tid < cache.size(); ++ tid) {
      for (int itx = 0; itx < cache[tid].size(); ++ itx) {
        if (create) {
          // std::cerr << "extract feature: create " << cache[tid][itx] << std::endl;
          model->space.retrieve(tid, cache[tid][itx], true);
        }

        int idx = model->space.index(tid, cache[tid][itx]);
        // std::cout << "key: " << cache[tid][itx] << " " << idx << std::endl;
        if (idx >= 0) { cache_again.push_back(idx); }
      }
    }

    int num_feat = cache_again.size();

    if (num_feat > 0 && ctx) {
      int l = 0;
      int * idx = new int[num_feat];
      for (int j = 0; j < num_feat; ++ j) {
        idx[j] = cache_again[j];
      }

      ctx->uni_features[pos][l] = new FeatureVector;
      ctx->uni_features[pos][l]->n = num_feat;
      ctx->uni_features[pos][l]->val = 0;
      ctx->uni_features[pos][l]->loff = 0;
      ctx->uni_features[pos][l]->idx = idx;

      for (l = 1; l < L; ++ l) {
        ctx->uni_features[pos][l] = new FeatureVector;
        ctx->uni_features[pos][l]->n = num_feat;
        ctx->uni_features[pos][l]->idx = idx;
        ctx->uni_features[pos][l]->val = 0;
        ctx->uni_features[pos][l]->loff = l;
      }
    }
  }
}

void Postagger::calculate_scores(const Instance* inst,  const DecodeContext* ctx,
    bool use_avg, ScoreMatrix* scm) const {
  int len = inst->size();
  int L = model->num_labels();

  scm->uni_scores.resize(len, L); scm->uni_scores = NEG_INF;
  scm->bi_scores.resize(L, L);    scm->bi_scores = NEG_INF;

  for (int i = 0; i < len; ++ i) {
    for (int l = 0; l < L; ++ l) {
      FeatureVector * fv = ctx->uni_features[i][l];
      if (!fv) { continue; }
      scm->uni_scores[i][l] = model->param.dot(ctx->uni_features[i][l], use_avg);
    }
  }

  for (int pl = 0; pl < L; ++ pl) {
    for (int l = 0; l < L; ++ l) {
      int idx = model->space.index(pl, l);
      scm->bi_scores[pl][l] = model->param.dot(idx, use_avg);
    }
  }
}

void Postagger::build_labels(const Instance * inst,
    std::vector<std::string>& tags) const {
  int len = inst->size();
  if (inst->predicted_tagsidx.size() != len) {
    return;
  }

  tags.resize(len);
  for (int i = 0; i < len; ++ i) {
    tags[i] = model->labels.at(inst->predicted_tagsidx[i]);
  }
}

}     //  end for namespace postagger
}     //  end for namespace ltp
