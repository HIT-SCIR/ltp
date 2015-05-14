#ifndef __LTP_FRAMEWORK_DECODER_H__
#define __LTP_FRAMEWORK_DECODER_H__

#include "utils/math/mat.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

namespace ltp {
namespace framework {

struct ViterbiLatticeItem {
  ViterbiLatticeItem (const size_t& _i, const size_t& _l, const double& _score,
      const ViterbiLatticeItem* _prev)
    : i(_i), l(_l), score(_score), prev(_prev) {}

  ViterbiLatticeItem (const size_t& _l, const double& _score)
    : i(0),  l(_l), score(_score), prev(0) {}

  size_t i;
  size_t l;
  double score;
  const ViterbiLatticeItem* prev;
};

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

    for (size_t i = 0; i < L; ++ i) {
      for (size_t t = 0; t < T; ++ t) {
        if (i == 0) {
          ViterbiLatticeItem* item = new ViterbiLatticeItem(i, t, scm.emit(i, t), NULL);
          lattice_insert(lattice[i][t], item);
        } else {
          for (size_t pt = 0; pt < T; ++ pt) {
            const ViterbiLatticeItem* prev = lattice[i-1][pt];
            if (!prev) { continue; }

            double s = scm.emit(i, t) + scm.tran(pt, t) + prev->score;
            ViterbiLatticeItem* item = new ViterbiLatticeItem(i, t, s, prev);
            lattice_insert(lattice[i][t], item);
          }
        }
      }
    }

    get_result(L-1, output);
    free_lattice();
  }

  void decode(const ViterbiScoreMatrix& scm,
      const ViterbiDecodeConstrain& con,
      std::vector<int>& output) {
    size_t L = scm.length();
    size_t T = scm.labels();

    init_lattice(L, T);

    for (size_t i = 0; i < L; ++ i) {
      for (size_t t = 0; t < T; ++ t) {
        if (!con.can_emit(i, t)) { continue; }

        if (i == 0) {
          ViterbiLatticeItem* item = new ViterbiLatticeItem(i, t, scm.emit(i, t), NULL);
          lattice_insert(lattice[i][t], item);
        } else {
          for (size_t pt = 0; pt < T; ++ pt) {
            if (!con.can_emit(i-1, pt) || !con.can_tran(pt, t)) { continue; }

            const ViterbiLatticeItem* prev = lattice[i-1][pt];
            if (!prev) { continue; }

            double s = scm.emit(i, t) + scm.tran(pt, t) + prev->score;
            ViterbiLatticeItem* item = new ViterbiLatticeItem(i, t, s, prev);
            lattice_insert(lattice[i][t], item);
          }
        }
      }
    }
    get_result(L-1, output);
    free_lattice();
  }
protected:
  void init_lattice(const size_t& L, const size_t& T) {
    lattice.resize(L, T);
    lattice = NULL;
  }

  void get_result(std::vector<int>& output) {
    size_t L = lattice.nrows();
    get_result(L- 1, output);
  }

  void get_result(const size_t& p, std::vector<int>& output) {
    size_t T = lattice.ncols();

    const ViterbiLatticeItem* best = NULL;
    for (size_t t = 0; t < T; ++ t) {
      if (!lattice[p][t]) {
        continue;
      }

      if (best == NULL || lattice[p][t]->score > best->score) {
        best = lattice[p][t];
      }
    }

    output.resize(p+1);
    while (best) {
      output[best->i] = best->l;
      best = best->prev;
    }
  }

  void free_lattice() {
    size_t L = lattice.total_size();
    const ViterbiLatticeItem ** p = lattice.c_buf();
    for (size_t i = 0; i < L; ++ i) {
      if (p[i]) {
        delete p[i];
        p[i] = 0;
      }
    }
  }

  void lattice_insert(const ViterbiLatticeItem* &position,
      const ViterbiLatticeItem * const item) {
    if (position == NULL) {
      position = item;
    } else if (position->score < item->score) {
      delete position;
      position = item;
    } else {
      delete item;
    }
  }

  math::Mat< const ViterbiLatticeItem * > lattice;
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_DECODER_H__
