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
      const math::SparseVec& correct_features,
      const math::SparseVec& predict_features,
      const size_t& timestamp,
      const size_t& margin,
      Model* model,
      math::SparseVec& update_features) {

    update_features.zero();
    if (algorithm == "pa") {
      update_features.add(correct_features, 1.);
      update_features.add(predict_features, -1.);

      double error = margin;
      double score = model->param.dot(update_features, false);
      double norm = update_features.L2();

      double step = 0.;
      if (norm < 1e-8) {
        step = 0;
      } else {
        step = (error - score) / norm;
      }
      model->param.add(update_features, timestamp, step);
    } else if (algorithm == "ap") {
      update_features.add(correct_features, 1.);
      update_features.add(predict_features, -1.);

      model->param.add(update_features, timestamp, 1.);
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
   *  @param[in]  model           The pointer to the model.
   *  @param[in]  threshold       The threshold for rare feature.
   *  @param[out] update_counts   The update time counts.
   */
  Model* erase_rare_features(Model* model, const size_t& threshold,
      const std::vector<size_t>& update_counts) {
    size_t nr_dicts = model->space.num_dicts();
    size_t T = model->labels.size();

    TRACE_LOG("trunc-model: number of feature group is %d", nr_dicts);
    Model* new_model = new Model(nr_dicts);
    // copy the label indexable map to the new model
    for (size_t t = 0; t < T; ++ t) {
      const char* key = model->labels.at(t);
      new_model->labels.push(key);
    }
    TRACE_LOG("trunc-model: building labels map for truncated model is done.");

    T = new_model->num_labels();
    new_model->space.set_num_labels(T);

    for (FeatureSpaceIterator itx = model->space.begin(); itx != model->space.end();
        ++ itx) {
      const char* key = itx.key();
      size_t tid = itx.tid();
      int id = model->space.index(tid, key);
      bool all_zero = true;
      for (size_t t = 0; t < T; ++ t) {
        double p = model->param.dot(id + t);
        if (p != 0.) { all_zero = false; break; }
      }

      if (all_zero) { continue; }

      size_t idx = model->space.retrieve(tid, key, false);
      if (idx < update_counts.size() && update_counts[idx] < threshold) { continue; }
      new_model->space.retrieve(tid, key, true);
    }

    TRACE_LOG("trunc-model: building new feature space for truncated model is done.");
    new_model->param.realloc(new_model->space.dim());
    TRACE_LOG("trunc-model: parameter dimension of new model is %d.",
        new_model->space.dim());

    for (FeatureSpaceIterator itx = new_model->space.begin();
        itx != new_model->space.end(); ++ itx) {
      const char* key = itx.key();
      size_t tid = itx.tid();
      int old_id = model->space.index(tid, key);
      int new_id = new_model->space.index(tid, key);

      for (size_t t = 0; t < T; ++ t) {
        // pay attention to this place, use average should be set true
        // some dirty code
        new_model->param._W[new_id+ t]      = model->param._W[old_id+ t];
        new_model->param._W_sum[new_id+ t]  = model->param._W_sum[old_id+ t];
        new_model->param._W_time[new_id+ t] = model->param._W_time[old_id+ t];
      }
    }

    for (int pt = 0; pt < T; ++ pt) {
      for (int t = 0; t < T; ++ t) {
        int old_id = model->space.index(pt, t);
        int new_id = new_model->space.index(pt, t);

        new_model->param._W[new_id]      = model->param._W[old_id];
        new_model->param._W_sum[new_id]  = model->param._W_sum[old_id];
        new_model->param._W_time[new_id] = model->param._W_time[old_id];
      }
    }
    TRACE_LOG("trunc-model: building parameter for new model is done.");
    TRACE_LOG("trunc-model: building new model is done.");
    return new_model;
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
