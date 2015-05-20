#ifndef __LTP_FRAMEWORK_FRONTEND_H__
#define __LTP_FRAMEWORK_FRONTEND_H__

#include "framework/model.h"
#include "utils/math/mat.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"
#include "utils/logging.hpp"

namespace ltp {
namespace framework {

enum FrontendMode {
  kLearn,
  kTest,
  kDump
};

class Frontend {
protected:
  FrontendMode mode;  //! The frontend model.
public:
  //! The constructor
  Frontend(const FrontendMode& _mode): mode(_mode) {}

  void learn_passive_aggressive(const math::SparseVec& updated_features,
    const size_t& timestamp, const double& error, Model* model) {
    double score = model->param.dot(updated_features, false);
    learn_passive_aggressive(updated_features, timestamp, error, score, model);
  }

  void learn_passive_aggressive(const math::SparseVec& updated_features,
    const size_t& timestamp, const double& error, const double& score, Model* model) {
    double norm = updated_features.L2();
    double step = 0.;

    if (norm < 1e-8) {
      step = 0;
    }
    else {
      step = (error - score) / norm;
    }
    model->param.add(updated_features, timestamp, step);
  }

  void learn_averaged_perceptron(const math::SparseVec& updated_features,
    const size_t& timestamp, Model* model) {
    model->param.add(updated_features, timestamp, 1.);
  }

  // The perceptron learning part.
  /**
   * Perform the online learning algorithm to learn the model parameters.
   *
   *  @param[in]  algorithm         The learning algorithm, can be "pa" or "ap".
   *  @param[in]  correct_features  The sparse vector of correct features.
   *  @param[in]  predict_features  The sparse vector of predict features.
   *  @param[in]  timestamp         The timestamp for this update.
   *  @param[in]  margin            The learning margin.
   *  @param[out] model             The model to be learnt.
   *  @param[out] update_features   The features the updated.
   */
  void learn(const std::string& algorithm,
      const math::SparseVec& updated_features,
      const size_t& timestamp,
      const double& margin,
      Model* model) {
    if (algorithm == "pa") {
      learn_passive_aggressive(updated_features, timestamp, margin, model);
    } else if (algorithm == "ap") {
      learn_averaged_perceptron(updated_features, timestamp, model);
    } else {
      WARNING_LOG("framework, frontend: unknown learning algorithm: %s", algorithm.c_str());
    }
  }

  // The model truncation part.
  /**
   * Increase the updated counter by group. For example, the feature "w0=was,t=NN"(1) and
   * "w0=was,t=VV"(2) is mapped into the same group and both the update on (1) and (2) are
   * counted.
   *
   *  @param[in]  model           The pointer to the model.
   *  @param[in]  vec             The sparse vector.
   *  @param[out] update_counts   The update time counts.
   */
  void increase_groupwise_update_counts(
      const Model* model,
      const math::SparseVec& vec,
      std::vector<size_t>& update_counts) {
    size_t T = model->num_labels();
    for (math::SparseVec::const_iterator itx = vec.begin();
        itx != vec.end(); ++ itx) {
      size_t fid = itx->first;
      size_t gid = fid / T;
      if (itx->second != 0.0 && gid < update_counts.size()) {
        ++ update_counts[gid];
      }
    }
  }

  /**
   * Erase the rarely updated features.
   *
   *  @param[in]  source          The pointer to the source model.
   *  @param[out] target          The pointer to the target model.
   *  @param[in]  threshold       The threshold for rare feature.
   *  @param[out] update_counts   The update time counts.
   */
  void erase_rare_features(const Model* source, Model* target, const size_t& threshold,
      const std::vector<size_t>& update_counts) {
    if (target && source->space.num_dicts() != target->space.num_dicts()) {
      ERROR_LOG("trunc-model: source and target should be initialized with same #dicts.");
      return;
    }
    size_t T = source->labels.size();

    // copy the label indexable map to the new model
    for (size_t t = 0; t < T; ++ t) {
      const char* key = source->labels.at(t);
      target->labels.push(key);
    }
    TRACE_LOG("trunc-model: building labels map for truncated model is done.");

    T = target->num_labels();
    target->space.set_num_labels(T);

    // Maintain the feature space.
    for (FeatureSpaceIterator itx = source->space.begin(); itx != source->space.end();
        ++ itx) {
      const char* key = itx.key();
      size_t tid = itx.tid();
      int id = source->space.index(tid, key);
      bool all_zero = true;
      for (size_t t = 0; t < T; ++ t) {
        double p = source->param.dot(id + t);
        if (p != 0.) { all_zero = false; break; }
      }

      if (all_zero) { continue; }
      int idx = source->space.retrieve(tid, key);
      if ((size_t)idx < update_counts.size() && update_counts[idx] < threshold) { continue; }
      target->space.retrieve(tid, key, true);
    }

    TRACE_LOG("trunc-model: building new feature space for truncated model is done.");
    target->param.realloc(target->space.dim());
    TRACE_LOG("trunc-model: parameter dimension of new model is %d.",
        target->space.dim());

    // Maintain the parameters
    for (FeatureSpaceIterator itx = target->space.begin();
        itx != target->space.end(); ++ itx) {
      const char* key = itx.key();
      size_t tid = itx.tid();
      int old_id = source->space.index(tid, key);
      int new_id = target->space.index(tid, key);

      for (size_t t = 0; t < T; ++ t) {
        target->param._W[new_id+ t]      = source->param._W[old_id+ t];
        target->param._W_sum[new_id+ t]  = source->param._W_sum[old_id+ t];
        target->param._W_time[new_id+ t] = source->param._W_time[old_id+ t];
      }
    }

    for (size_t pt = 0; pt < T; ++ pt) {
      for (size_t t = 0; t < T; ++ t) {
        int old_id = source->space.index(pt, t);
        int new_id = target->space.index(pt, t);

        target->param._W[new_id]      = source->param._W[old_id];
        target->param._W_sum[new_id]  = source->param._W_sum[old_id];
        target->param._W_time[new_id] = source->param._W_time[old_id];
      }
    }

    // last, we need to care about the timestamp.
    target->param._last_timestamp = source->param._last_timestamp;
    TRACE_LOG("trunc-model: building parameter for new model is done.");
    TRACE_LOG("trunc-model: building new model is done.");
  }

  // Utility
  void collect_features(const Model* model,
      const math::Mat<math::FeatureVector*>& features,
      const std::vector<int>& tagsidx, math::SparseVec& vec) {
    vec.zero();
    for (size_t i = 0; i < tagsidx.size(); ++ i) {
      int l = tagsidx[i];
      const math::FeatureVector* fv = features[i][l];

      if (!fv) {
        continue;
      }
      vec.add(fv->idx, fv->val, fv->n, fv->loff, 1.);
      if (i > 0) {
        int pl = tagsidx[i-1];
        int idx = model->space.index(pl, l);
        vec.add(idx, 1.);
      }
    }
  }
};

}   //  namespace framework
}   //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_FRONTEND_H__
