#ifndef __LTP_PARSER_INSTANCE_H__
#define __LTP_PARSER_INSTANCE_H__

#include <iostream>
#include <string>
#include <set>

#include "utils/math/mat.h"
#include "utils/math/sparsevec.h"
#include "utils/math/featurevec.h"

namespace ltp {
namespace parser {

using namespace ltp::math;

class Instance {
public:
  Instance(){
  }

  ~Instance() {
    free_features();
  }

  /*
   * Get length of the instance. Instance's length is defined as number
   * of form tokens.
   *
   *  @return   size_t        the number of tokens
   */
  size_t size() const {
    return forms.size();
  }

  /*
   * Get number of error of the heads. Prepositive condition is number
   * of predicted_heads greater than 0 and number of predicated_heads
   * equals number of heads.
   *
   *  @param[in]  ignore_punctation   specify whether ignore punction
   *  @return   int         the number of errors
   */
  int num_error_heads(bool ignore_punctation = true) const {
    if (predicted_heads.size() == 0) {
      return size();
    }

    int ret = 0;
    int len = size();
    for (int i = 1; i < len; ++ i) {
      if (ignore_punctation && postags[i] == "wp") {
        continue;
      }

      if (predicted_heads[i] != heads[i]) {
        ++ ret;
      }
    }

    return ret;
  }

  int num_error_labels(bool ignore_punctation = true) {
    if (predicted_heads.size() == 0 || predicted_deprelsidx.size() == 0) {
      return size();
    }

    int ret = 0;
    int len = size();
    for (int i = 1; i < len; ++ i) {
      if (ignore_punctation && postags[i] == "wp") {
        continue;
      }

      if (predicted_heads[i] == heads[i]
          && predicted_deprelsidx[i] != deprelsidx[i]) {
        ++ ret;
      }
    }

    return ret;
  }

  double num_errors() {
    return num_error_heads() + 0.5 * num_error_labels();
  }

  int num_correct_heads(bool ignore_punctation = true) {
    if (predicted_heads.size() == 0) {
      return 0;
    }

    int ret = 0;
    int len = size();
    for (int i = 1; i < len; ++ i) {
      if (ignore_punctation && postags[i] == "wp") {
        continue;
      }

      if (predicted_heads[i] == heads[i]) {
        ++ ret;
      }
    }

    return ret;
  }

  int num_correct_heads_and_labels(bool ignore_punctation = true) {
    if (predicted_heads.size() == 0 || predicted_deprelsidx.size() == 0) {
      return 0;
    }

    int ret = 0;
    int len = size();
    for (int i = 1; i < len; ++ i) {
      if (ignore_punctation && postags[i] == "wp") {
        continue;
      }

      if (predicted_heads[i] == heads[i]
          && (predicted_deprelsidx[i] == deprelsidx[i])) {
        ++ ret;
      }
    }

    return ret;
  }

  int num_rels(bool ignore_punctation = true) const {
    if (!ignore_punctation) {
      return forms.size() - 1;
    } else {
      int ret = 0;
      int len = size();
      for (int i = 1; i < len; ++ i) {
        if (postags[i] != "wp") {
          ++ ret;
        }
      }
      return ret;
    }
    return -1;
  }

  void cleanup() {
    free_features();
    features.zero();
    predicted_features.zero();
    depu_features.dealloc();
    depu_scores.dealloc();
    depl_features.dealloc();
    depl_scores.dealloc();
    sibu_features.dealloc();
    sibu_scores.dealloc();
    sibl_features.dealloc();
    sibl_scores.dealloc();
    grdu_features.dealloc();
    grdu_scores.dealloc();
    grdl_features.dealloc();
    grdl_scores.dealloc();
  }
public:

  std::vector< std::string >                forms;    /*< the forms */
  std::vector< std::string >                lemmas;   /*< the lemmas */
  std::vector< std::vector< std::string> >  chars;    /*< the characters */
  std::vector< std::string >                postags;  /*< the postags */

  std::vector<int>              heads;
  std::vector<int>              deprelsidx;
  std::vector< std::string >    deprels;
  std::vector<int>              predicted_heads;
  std::vector<int>              predicted_deprelsidx;
  std::vector< std::string >    predicted_deprels;

  SparseVec         predicted_features;
  SparseVec         features;

  /* features group */
  //Vec<FeatureVector *>  posu_features;
  //Vec<double>       posu_scores;

  Mat<FeatureVector *>  depu_features;
  Mat<double>           depu_scores;

  Mat3<FeatureVector *> depl_features;
  Mat3<double>          depl_scores;

  Mat3<FeatureVector *> sibu_features;
  Mat3<double>          sibu_scores;

  Mat4<FeatureVector *> sibl_features;
  Mat4<double>          sibl_scores;

  Mat3<FeatureVector *> grdu_features;
  Mat3<double>          grdu_scores;

  Mat4<FeatureVector *> grdl_features;
  Mat4<double>          grdl_scores;

  std::vector<int>      verb_cnt;
  std::vector<int>      conj_cnt;
  std::vector<int>      punc_cnt;

private:
  void free_features() {
    int len;
    FeatureVector ** fvs;
    if ((len = depu_features.total_size()) > 0
        && (fvs = depu_features.c_buf())) {
      for (int i = 0; i < len; ++ i) {
        if (fvs[i]) {
          fvs[i]->clear();
          delete fvs[i];
        }
      }
    }

    // in labeled case, different labels share memory, index should
    // be avoid double delete
    if ((len = depl_features.total_size()) > 0 
        && (fvs = depl_features.c_buf())) {
      int d1 = depl_features.dim1();
      int d2 = depl_features.dim2();
      int d3 = depl_features.dim3();

      for (int i = 0; i < d1; ++ i) {
        for (int j = 0; j < d2; ++ j) {
          if (depl_features[i][j][0]) {
            depl_features[i][j][0]->clear();
          }
          for (int l = 0; l < d3; ++ l) {
            if (depl_features[i][j][l]) {
              delete depl_features[i][j][l];
            }
          }
        }
      }
    }

    if ((len = sibu_features.total_size()) > 0
        && (fvs = sibu_features.c_buf())) {
      for (int i = 0; i < len; ++ i) {
        if (fvs[i]) {
          fvs[i]->clear();
          delete fvs[i];
        }
      }
    }

    if ((len = sibl_features.total_size()) > 0
        && (fvs = sibl_features.c_buf())) {
      int d1 = sibl_features.dim1();
      int d2 = sibl_features.dim2();
      int d3 = sibl_features.dim3();
      int d4 = sibl_features.dim4();

      for (int i = 0; i < d1; ++ i) {
        for (int j = 0; j < d2; ++ j) {
          for (int k = 0; k < d3; ++ k) {
            if (sibl_features[i][j][k][0]) {
              sibl_features[i][j][k][0]->clear();
            }
            for (int l = 0; l < d4; ++ l) {
              if (sibl_features[i][j][k][l]) {
                delete sibl_features[i][j][k][l];
              }
            }
          }
        }
      }
    }

    if ((len = grdu_features.total_size()) > 0
        && (fvs = grdu_features.c_buf())) {
      for (int i = 0; i < len; ++ i) {
        if (fvs[i]) {
          fvs[i]->clear();
          delete fvs[i];
        }
      }
    }

    if (grdl_features.total_size() > 0
        && (fvs = grdl_features.c_buf())) {
      int d1 = grdl_features.dim1();
      int d2 = grdl_features.dim2();
      int d3 = grdl_features.dim3();
      int d4 = grdl_features.dim4();

      for (int i = 0; i < d1; ++ i) {
        for (int j = 0; j < d2; ++ j) {
          for (int k = 0; k < d3; ++ k) {
            if (grdl_features[i][j][k][0]) {
              grdl_features[i][j][k][0]->clear();
            }
            for (int l = 0; l < d4; ++ l) {
              if (grdl_features[i][j][k][l]) {
                delete grdl_features[i][j][k][l];
              }
            }
          }
        }
      }
    }

  }
};  // end for class Instance
}   // end for namespace parser
}   // end for namespace ltp

#endif  // end for __INSTANCE_H__
