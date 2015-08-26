#ifndef __LTP_FRAMEWORK_DECODER_H__
#define __LTP_FRAMEWORK_DECODER_H__

#include "utils/math/mat.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"
#include <cmath>
#include <float.h>

namespace ltp {
namespace framework {

class ViterbiDecodeConstrain {
public:
  virtual bool can_emit(const size_t& i, const size_t& j) const {
    return true;
  }

  virtual bool can_tran(const size_t& i, const size_t& j) const {
    return true;
  }
};

class ViterbiFeatureContext {
public:
  math::SparseVec correct_features;   //! the gold features.
  math::SparseVec predict_features;   //! the predicted features.
  math::Mat<math::FeatureVector*> uni_features;  //! the feature cache.

  ViterbiFeatureContext() {}
  ~ViterbiFeatureContext() { clear(); }

  void clear() {
    if (uni_features.total_size() > 0) {
      size_t d1 = uni_features.nrows();
      size_t d2 = uni_features.ncols();
      for (size_t i = 0; i < d1; ++ i) {
        if (uni_features[i][0]) {
          uni_features[i][0]->clear();
        }
        for (size_t j = 0; j < d2; ++j) {
          if (uni_features[i][j]) {
            delete uni_features[i][j];
          }
        }
      }
    }

    uni_features.dealloc();
    correct_features.zero();
    predict_features.zero();
  } // end clear
};

class ViterbiScoreMatrix {
private:
  math::Mat< double > emit_scores;
  math::Mat< double > tran_scores;
public:
  ViterbiScoreMatrix() {}
  ViterbiScoreMatrix(const size_t& L, const size_t& T) {
    emit_scores.resize(L, T);
    tran_scores.resize(T, T);
  }

  ~ViterbiScoreMatrix() {}

  void clear() {
    emit_scores.dealloc();
    tran_scores.dealloc();
  }

  void resize(const size_t& L, const size_t& T) {
    emit_scores.resize(L, T);
    tran_scores.resize(T, T);
  }

  void resize(const size_t& L, const size_t& T, const double& V) {
    emit_scores.resize(L, T); emit_scores = V;
    tran_scores.resize(T, T); tran_scores = V;
  }

  size_t labels() const {
    return emit_scores.ncols();
  }

  size_t length() const {
    return emit_scores.nrows();
  }

  double emit(const size_t& i, const size_t& j) const {
    return emit_scores[i][j];
  }

  double tran(const size_t& i, const size_t& j) const {
    return tran_scores[i][j];
  }

  double safe_emit(const size_t& i, const size_t& j,
      const double& default_retval = 0.) const {
    if (i >= emit_scores.nrows() || j >= tran_scores.ncols()) {
      return default_retval;
    }
    return emit_scores[i][j];
  }

  double safe_tran(const size_t& i, const size_t& j,
      const double& default_retval = 0.) const {
    if (i >= tran_scores.nrows() || j >= tran_scores.ncols()) {
      return default_retval;
    }
    return tran_scores[i][j];
  }

  void set_emit(const size_t& i, const size_t& j, const double& score) {
    emit_scores[i][j] = score;
  }

  void set_tran(const size_t& i, const size_t& j, const double& score) {
    tran_scores[i][j] = score;
  }

  void safe_set_emit(const size_t& i, const size_t& j, const double& score) {
    if (i >= emit_scores.nrows() || j >= tran_scores.ncols()) {
      return;
    }
    emit_scores[i][j] = score;
  }

  void safe_set_tran(const size_t& i, const size_t& j, const double& score) {
    if (i >= tran_scores.nrows() || j >= tran_scores.ncols()) {
      return;
    }
    tran_scores[i][j] = score;
  }
};

class ViterbiDecoder {
public:
  void decode(const ViterbiScoreMatrix& scm, std::vector<int>& output) {
    size_t L = scm.length();
    size_t T = scm.labels();

    init_lattice(L, T);

    for (size_t t = 0; t < T; ++t) {
      state[0][t] = scm.emit(0, t);
    }

    double best = -DBL_MAX;
    for (size_t i = 1; i < L; ++ i) {
      for (size_t t = 0; t < T; ++ t) {
        best = -DBL_MAX;
        for (size_t pt = 0; pt < T; ++ pt) {
          double s = state[i-1][pt] + scm.tran(pt, t);
          if (s > best) {
            best = s;
            back[i][t] = pt;
          }
        }
        state[i][t] = best + scm.emit(i, t);
      }
    }

    get_result(output);
  }

  void decode(const ViterbiScoreMatrix& scm,
      const ViterbiDecodeConstrain& con,
      std::vector<int>& output) {

    size_t L = scm.length();
    size_t T = scm.labels();

    init_lattice(L, T);

    for (size_t t = 0; t < T; ++t) {
      if (!con.can_emit(0, t)) continue;
      state[0][t] = scm.emit(0, t);
    }

    double best = -DBL_MAX;
    for (size_t i = 1; i < L; ++ i) {
      for (size_t t = 0; t < T; ++ t) {
        if (!con.can_emit(i, t)) continue;
        best = -DBL_MAX;
        for (size_t pt = 0; pt < T; ++ pt) {
          if (!con.can_emit(i-1, pt) || !con.can_tran(pt, t)) continue;
          double s = state[i-1][pt] + scm.tran(pt, t);
          if (s > best) {
            best = s;
            back[i][t] = pt;
          }
        }
        state[i][t] = best + scm.emit(i, t);
      }
    }

    get_result(output);
  }

protected:
  void init_lattice(const size_t& L, const size_t& T) {
    back.resize(L, T);
    back = -1;

    state.resize(L, T);
    state = -DBL_MAX;
  }

  void get_result(std::vector<int>& output) {
    size_t L = back.nrows();
    get_result(L-1, output);
  }

  void get_result(const size_t& p, std::vector<int>& output) {
    size_t T = back.ncols();

    output.resize(p+1);
    double best = -DBL_MAX;

    for (size_t t = 0; t < T; ++t) {
      double s = state[p][t];
      if (s > best) {
        best = s;
        output[p] = t;
      }
    }

    for (int i = p-1; i >= 0; --i) {
      output[i] = back[i+1][output[i+1]];
    }
  }


  math::Mat<int>     back;
  math::Mat<double>  state;

};


class ViterbiDecoderWithMarginal : public ViterbiDecoder {
public:
  ViterbiDecoderWithMarginal(): sequence_prob(false), marginal_prob(false) {}

  void decode(const ViterbiScoreMatrix& scm, std::vector<int>& output) {
    ViterbiDecoder::decode(scm, output);
  }

  void decode(const ViterbiScoreMatrix& scm,
              const ViterbiDecodeConstrain& con,
              std::vector<int>& output) {
    ViterbiDecoder::decode(scm, con, output);
  }

  void init_prob_ctx(const ViterbiScoreMatrix& scm,
          bool avg = false,
          size_t last_timestamp = 1) {
    init_exp(scm, avg, last_timestamp);
    calc_alpha_score();
    calc_beta_score();
  }

  void init_prob_ctx(const ViterbiScoreMatrix& scm,
          const ViterbiDecodeConstrain& con,
          bool avg = false,
          size_t last_timestamp = 1) {
    init_exp(scm, avg, last_timestamp);
    calc_alpha_score(con);
    calc_beta_score(con);
  }

  void calc_sequence_probability(const std::vector<int>& path, double& sequence_probability) {
    sequence_probability = marginal_path(path, 0, path.size());
  }

  void calc_point_probabilities(const std::vector<int>& path, std::vector<double>& point_probabilities) {
    size_t len = path.size();
    point_probabilities.resize(len);
    for (size_t i = 0; i < len; ++i) {
      point_probabilities[i] = marginal_point(i, path[i]);
    }
  }

  void calc_partial_probabilities(const std::vector<int>& path,
          const std::vector<int>& partial_idx,
          std::vector<double>& partial_probabilities) {

      partial_probabilities.resize(partial_idx.size());
      for (size_t i = 0; i < partial_idx.size() - 1; ++i) {
        if ( i == partial_idx.size() - 1) {
          partial_probabilities[i] = marginal_path(path, partial_idx[i], path.size());
        } else {
          partial_probabilities[i] = marginal_path(path, partial_idx[i], partial_idx[i+1]);
        }
      }
  }

  void set_sequence_prob(bool flag) {
    sequence_prob = flag;
  }

  void set_marginal_prob(bool flag) {
    marginal_prob = flag;
  }


protected:
  void init_exp(const ViterbiScoreMatrix& scm, bool avg = false, size_t last_timestamp = 1) {
    size_t L = scm.length();
    size_t T = scm.labels();

    exp_emit.resize(L, T);

    if (!avg) {
        last_timestamp = 1;
    }

    for (int i = 0; i < L; ++ i) {
      for (int t = 0; t < T; ++ t) {
        exp_emit[i][t] = exp(scm.emit(i, t) / last_timestamp);
      }
    }

    exp_tran.resize(T, T);
    for (int i = 0; i < T; ++ i) {
      for (int j = 0; j < T; ++ j) {
        exp_tran[i][j] = exp(scm.tran(i, j) / last_timestamp);
      }
    }
  }

  void calc_alpha_score(void) {
    size_t L = exp_emit.nrows();
    size_t T = exp_emit.ncols();

    alpha_score.resize(L, T);
    alpha_score = 0.;
    scale.resize(L);

    for (size_t j = 0 ; j < T; ++j) {
      alpha_score[0][j] = exp_emit[0][j];
    }
    double sum = row_sum(alpha_score, 0);
    scale[0] = (sum == 0.) ? 1. : 1. / sum;
    row_scale(alpha_score, 0, scale[0]);

    for (size_t i = 1; i < L; ++i) {
      for (size_t t = 0; t < T; ++t) {
        for (size_t pt = 0;pt < T; ++pt) {
          alpha_score[i][t] += alpha_score[i-1][pt] * exp_tran[pt][t];
        }
        alpha_score[i][t] *= exp_emit[i][t];
      }
      sum = row_sum(alpha_score, i);
      scale[i] = (sum == 0.) ? 1. : 1. / sum;
      row_scale(alpha_score, i, scale[i]);
    }
  }

  void calc_beta_score(void) {
    size_t L = exp_emit.nrows();
    size_t T = exp_emit.ncols();

    beta_score.resize(L, T);
    beta_score = 0.;

    for (size_t j = 0; j < T; ++j) {
      beta_score[L-1][j] = scale[L-1];
    }

    double * tmp_row = new double[T];
    for (int i = L - 2; i >= 0; --i) {
      for (size_t nt = 0; nt < T; ++nt) {
        tmp_row[nt] = beta_score[i+1][nt] * exp_emit[i+1][nt];
      }
      for (size_t t = 0; t < T; ++t) {
        for (size_t nt = 0; nt < T; ++nt) {
          beta_score[i][t] += tmp_row[nt] * exp_tran[t][nt];
        }
      }
      row_scale(beta_score, i, scale[i]);
    }
    delete[] tmp_row;
  }

  void calc_alpha_score(const ViterbiDecodeConstrain& con) {
    size_t L = exp_emit.nrows();
    size_t T = exp_emit.ncols();

    alpha_score.resize(L, T);
    alpha_score = 0.;
    scale.resize(L);

    for (size_t j = 0 ; j < T; ++j) {
      if (!con.can_emit(0, j)) { continue; }
      alpha_score[0][j] = exp_emit[0][j];
    }
    double sum = row_sum(alpha_score, 0, con);
    scale[0] = (sum == 0.) ? 1. : 1. / sum;
    row_scale(alpha_score, 0, scale[0], con);

    for (size_t i = 1; i < L; ++i) {
      for (size_t t = 0; t < T; ++t) {
        if (!con.can_emit(i, t)) { continue; }
        for (size_t pt = 0; pt < T; ++pt) {
          if (!con.can_emit(i-1, pt) || !con.can_tran(pt, t)) { continue; }
          alpha_score[i][t] += alpha_score[i-1][pt] * exp_tran[pt][t];
        }
        alpha_score[i][t] *= exp_emit[i][t];
      }
      sum = row_sum(alpha_score, i, con);
      scale[i] = (sum == 0.) ? 1. : 1. / sum;
      row_scale(alpha_score, i, scale[i], con);
    }
  }

  void calc_beta_score(const ViterbiDecodeConstrain& con) {
    size_t L = exp_emit.nrows();
    size_t T = exp_emit.ncols();

    beta_score.resize(L, T);
    beta_score = 0.;


    for (size_t j = 0; j < T; ++j) {
      if (!con.can_emit(L-1, j)) { continue; }
      beta_score[L-1][j] = scale[L-1];
    }

    double * tmp_row = new double[T];
    for (int i = L - 2; i >= 0; --i) {
      for (size_t nt = 0; nt < T; ++nt) {
        if (!con.can_emit(i+1, nt)) { continue; }
        tmp_row[nt] = beta_score[i+1][nt] * exp_emit[i+1][nt];
      }
      for (size_t t = 0; t < T; ++t) {
        if (!con.can_emit(i, t)) { continue; }
        for (size_t nt = 0; nt < T; ++nt) {
          if (!con.can_emit(i+1, nt) || !con.can_tran(t, nt)) { continue; }
          beta_score[i][t] += tmp_row[nt] * exp_tran[t][nt];
        }
      }
      row_scale(beta_score, i, scale[i], con);
    }
    delete[] tmp_row;
  }

  double row_sum(const math::Mat<double>& mat, size_t i) const {
    double sum = 0.;
    for (size_t j = 0; j < mat.ncols(); ++j) {
      sum += mat[i][j];
    }
    return sum;
  }

  void row_scale(math::Mat<double>& mat, size_t i, double scale) {
    for (size_t j = 0 ; j < mat.ncols(); ++j) {
      mat[i][j] *= scale;
    }
  }

  double row_sum(const math::Mat<double>& mat,
          size_t i,
          const ViterbiDecodeConstrain& con) const {
    double sum = 0.;
    for (size_t j = 0; j < mat.ncols(); ++j) {
      if (!con.can_emit(i, j)) { continue; }
      sum += mat[i][j];
    }
    return sum;
  }

  void row_scale(math::Mat<double>& mat,
          size_t i,
          double scale,
          const ViterbiDecodeConstrain& con) {
    for (size_t j = 0 ; j < mat.ncols(); ++j) {
      if (!con.can_emit(i, j)) { continue; }
      mat[i][j] *= scale;
    }
  }

  double marginal_point(size_t i, size_t j) const {
    return alpha_score[i][j] * beta_score[i][j] / scale[i];
  }

  double marginal_path(const std::vector<int>& path, size_t begin, size_t end) const {
    double prob = alpha_score[begin][path[begin]] * beta_score[end-1][path[end-1]] / scale[begin];

    for (size_t i = begin; i < end-1; ++i) {
      prob *= (exp_tran[path[i]][path[i+1]] * exp_emit[i+1][path[i+1]] * scale[i]);
    }

    return prob;
  }



protected:

   math::Mat<double>     exp_emit;
   math::Mat<double>     exp_tran;
   math::Mat<double>     alpha_score;
   math::Mat<double>     beta_score;
   std::vector<double>   scale;
   bool                  sequence_prob;
   bool                  marginal_prob;

};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_DECODER_H__
