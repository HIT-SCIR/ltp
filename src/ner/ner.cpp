#include "utils/logging.hpp"
#include "ner/ner.h"
#include "ner/instance.h"
#include "ner/extractor.h"
#include "ner/options.h"
#include "ner/settings.h"

#include <iostream>
#include <sstream>
#include <fstream>

namespace ltp {
namespace ner {

using framework::ViterbiFeatureContext;
using framework::ViterbiScoreMatrix;
using utility::StringVec;
using math::FeatureVector;

const std::string NamedEntityRecognizer::model_header = "otner";
const std::string NamedEntityRecognizer::delimiter = "-";

NamedEntityRecognizer::NamedEntityRecognizer(): model(0), glob_con(0) {}
NamedEntityRecognizer::~NamedEntityRecognizer() {
  if (model) { delete model; model = 0; }
  if (glob_con) { delete glob_con;  glob_con = 0; }
}

void NamedEntityRecognizer::build_glob_tran_cons(
    const std::unordered_set<std::string>& ne_types) {
  if (glob_con != NULL) {
    WARNING_LOG("Transition constrain should not be double allocated.");
  }

  const std::string& _ = delimiter;
  std::vector<std::string> includes;
  includes.push_back("O -> O");

  std::stringstream S;
  for (std::unordered_set<std::string>::const_iterator i = ne_types.begin();
      i != ne_types.end(); ++ i) {
    S.str(""); S << "O -> S" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "O -> B" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "S" << _ << (*i) << " -> O"; includes.push_back(S.str());
    S.str(""); S << "E" << _ << (*i) << " -> O"; includes.push_back(S.str());
    S.str(""); S << "B" << _ << (*i) << " -> I" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "B" << _ << (*i) << " -> E" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "I" << _ << (*i) << " -> I" << _ << (*i); includes.push_back(S.str());
    S.str(""); S << "I" << _ << (*i) << " -> E" << _ << (*i); includes.push_back(S.str());

    for (std::unordered_set<std::string>::const_iterator j = ne_types.begin();
        j != ne_types.end(); ++ j) {
      S.str(""); S << "S" << _ << (*i) << " -> S" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "S" << _ << (*i) << " -> B" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "E" << _ << (*i) << " -> S" << _ << (*j); includes.push_back(S.str());
      S.str(""); S << "E" << _ << (*i) << " -> B" << _ << (*j); includes.push_back(S.str());
    }
  }

  //INFO_LOG("build-config: add %d constrains.", includes.size());
  //INFO_LOG("report: number of labels %d", model->num_labels());
  glob_con = new NERTransitionConstrain(model->labels, includes);
}


void NamedEntityRecognizer::extract_features(const Instance& inst,
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
        ctx->uni_features[pos][t]->val = 0;
        ctx->uni_features[pos][t]->loff = t;
        ctx->uni_features[pos][t]->idx = idx;
      }
    }
  }
}

void NamedEntityRecognizer::calculate_scores(const Instance& inst,
    const ViterbiFeatureContext& ctx, bool avg,
    ViterbiScoreMatrix* scm) const {
  size_t L = inst.size();
  size_t T = model->num_labels();

  scm->resize(L, T, -1e20);

  for (size_t i = 0; i < L; ++ i) {
    for (size_t t = 0; t < T; ++ t) {
      FeatureVector* fv = ctx.uni_features[i][t];
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

void NamedEntityRecognizer::build_entities(const Instance* inst,
    const std::vector<int>& tagsidx,
    std::vector<std::string>& entities,
    std::vector<std::string>& entities_tags,
    const size_t& delim_len) const {
  entities.clear();
  entities_tags.clear();

  size_t L= inst->size();
  if (L == 0) {
    return;
  }

  // should check the tagsidx size
  std::string entity = inst->raw_forms[0];
  std::string entity_tag = model->labels.at(tagsidx[0]);

  for (size_t i = 1; i < L; ++ i) {
    size_t tag = tagsidx[i];
    if (tag == 0 ||  //
        ((tag-1) % __num_pos_types__) == 0 ||
        ((tag-1) % __num_pos_types__) == 3) {
      entities.push_back(entity);
      entities_tags.push_back(entity_tag);

      entity = inst->raw_forms[i];
      if (tag == 0) { entity_tag = OTHER; }
      else {
        entity_tag = model->labels.at(tag);
        entity_tag = entity_tag.substr(1+delim_len); // Tag in format of S-xxx
      }
    } else {
      entity += inst->raw_forms[i];
    }
  }
  entities.push_back(entity);
  entities_tags.push_back(entity_tag);
}

}     //  end for namespace ner
}     //  end for namespace ltp
