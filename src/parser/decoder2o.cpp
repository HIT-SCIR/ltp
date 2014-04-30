#include "parser/decoder2o.h"
#include "parser/options.h"

namespace ltp {
namespace parser {

// ================================================================ //
// Decoder using dependency and sibling features                    //
// ================================================================ //

void Decoder2O::init_lattice(const Instance * inst) {
  int len = inst->size();
  _lattice_cmp.resize(len, len);
  _lattice_sib.resize(len, len);
  _lattice_incmp.resize(len, len);

  for (int i = 0; i < len; ++ i) {
    for (int j = 0; j < len; ++ j) {
      _lattice_cmp[i][j] = 0;
      _lattice_sib[i][j] = 0;
      _lattice_incmp[i][j] = 0;
    }
  }
  for (int i = 0; i < len; ++ i) {
    _lattice_cmp[i][i] = new LatticeItem(i);
  }
}

void Decoder2O::decode_projective(const Instance * inst) {
  int len = inst->size();
  for (int width = 1; width < len; ++ width) {
    for (int s = 0; s + width < len; ++ s) {
      int t = s + width;

      for (int l = 0; l < L; ++ l) {
        double shared_score = 0.;

        if (feat_opt.use_unlabeled_dependency) {
          shared_score += inst->depu_scores[s][t];
        }

        if (feat_opt.use_labeled_dependency) {
          shared_score += inst->depl_scores[s][t][l];
        }

        {   // I(s,t) = C(s,s) + C(t,s+1)
          const LatticeItem * const left  = _lattice_cmp[s][s];
          const LatticeItem * const right = _lattice_cmp[t][s + 1];

          if (!left || !right) {
            continue;
          }

          double score = left->_prob + right->_prob + shared_score;
          if (feat_opt.use_unlabeled_sibling) {
            score += inst->sibu_scores[s][t][s];
          }

          if (feat_opt.use_labeled_sibling) {
            score += inst->sibl_scores[s][t][s][l];
          }

          const LatticeItem * const item = new LatticeItem(INCMP,
                                                           s,
                                                           t,
                                                           score,
                                                           left,
                                                           right,
                                                           l);

          lattice_insert(_lattice_incmp[s][t], item);
        }   //  end for I(s,t) = C(s,s) + C(t,s+1)

        {   // I(s,t) = I(s,r) + S(r,t)
          for (int r = s + 1; r < t; ++ r) {
            const LatticeItem * const left = _lattice_incmp[s][r];
            const LatticeItem * const right = _lattice_sib[r][t];

            if (!left || !right) {
              continue;
            }

            double score = left->_prob + right->_prob + shared_score;

            if (feat_opt.use_unlabeled_sibling) {
              score += inst->sibu_scores[s][t][r];
            }

            if (feat_opt.use_labeled_sibling) {
              score += inst->sibl_scores[s][t][r][l];
            }

            const LatticeItem * const item = new LatticeItem(INCMP,
                                                             s,
                                                             t,
                                                             score,
                                                             left,
                                                             right,
                                                             l);

            lattice_insert(_lattice_incmp[s][t], item);
          }
        }   //  end for I(s,t) = I(s,r) + S(r,t)

      }   // end for for (l = 0; l < L; ++ l)

      if (s != 0) {   //  I(t,s) = C(s, t-1) + C(t, t)
        for (int l = 0; l < L; ++ l) {
          double shared_score = 0.;

          if (feat_opt.use_unlabeled_dependency) {
            shared_score += inst->depu_scores[t][s];
          }

          if (feat_opt.use_labeled_dependency) {
            shared_score += inst->depl_scores[t][s][l];
          }

          {   //  I(t,s) = C(s,t-1) + C(t,t)
            const LatticeItem * const left = _lattice_cmp[s][t-1];
            const LatticeItem * const right = _lattice_cmp[t][t];

            if (!left || !right) {
              continue;
            }

            double score =  left->_prob + right->_prob + shared_score;

            if (feat_opt.use_unlabeled_sibling) {
              score += inst->sibu_scores[t][s][t];
            }

            if (feat_opt.use_labeled_sibling) {
              score += inst->sibl_scores[t][s][t][l];
            }

            const LatticeItem * const item = new LatticeItem(INCMP,
                                                             t,
                                                             s,
                                                             score,
                                                             left,
                                                             right,
                                                             l);

            lattice_insert(_lattice_incmp[t][s], item);
          }   //  end for I(t, s) = C(s,t-1) + C(t,t)

          {   //  I(t,s) = S(s,r) + I(t,r)
            for (int r = s + 1; r < t; ++ r) {
              const LatticeItem * const left = _lattice_sib[s][r];
              const LatticeItem * const right = _lattice_incmp[t][r];

              if (!left || !right) {
                continue;
              }

              double score = left->_prob + right->_prob + shared_score;

              if (feat_opt.use_unlabeled_sibling) {
                score += inst->sibu_scores[t][s][r];
              }

              if (feat_opt.use_labeled_sibling) {
                score += inst->sibl_scores[t][s][r][l];
              }

              const LatticeItem * const item = new LatticeItem(INCMP,
                                                               t,
                                                               s,
                                                               score,
                                                               left,
                                                               right,
                                                               l);

              lattice_insert(_lattice_incmp[t][s], item);
            }
          }   //  end for I(t,s) = S(s,r) + I(t,r)

        }
      }   //  end for if (s != 0)

      {   // S(s,t) = C(s,r) + C(t,r+1)
        for (int r = s; r < t; ++ r) {
          const LatticeItem * const left = _lattice_cmp[s][r];
          const LatticeItem * const right = _lattice_cmp[t][r+1];

          if (!left || !right) {
            continue;
          }

          double score = left->_prob + right->_prob;

          const LatticeItem * const item = new LatticeItem(SIBSP,
              s,
              t,
              score,
              left,
              right);

          lattice_insert(_lattice_sib[s][t], item);
        }
      }   // end for S(s,t) = C(s,t) + C(t,r+1)

      {   //  C(s,t) = I(s,r) + C(r,t)
        for (int r = s + 1; r <= t; ++ r) {
          const LatticeItem * const left = _lattice_incmp[s][r];
          const LatticeItem * const right = _lattice_cmp[r][t];

          if (!left || !right) {
            continue;
          }

          double score = left->_prob + right->_prob;

          if (feat_opt.use_last_sibling) {
            if (feat_opt.use_unlabeled_sibling) {
              score += inst->sibu_scores[s][r][r];
            }

            if (feat_opt.use_labeled_sibling) {
              int l = left->_label_s_t;
              score += inst->sibl_scores[s][r][r][l];
            }
          }

          const LatticeItem * const item = new LatticeItem(CMP,
                                                           s,
                                                           t,
                                                           score,
                                                           left,
                                                           right);

          lattice_insert(_lattice_cmp[s][t], item);

        }
      }

      if (s != 0) {
        for (int r = s; r < t; ++ r) {
          const LatticeItem * const left = _lattice_cmp[r][s];
          const LatticeItem * const right = _lattice_incmp[t][r];

          if (!left || !right) {
            continue;
          }

          double score = left->_prob + right->_prob;

          if (feat_opt.use_last_sibling) {
            if (feat_opt.use_unlabeled_sibling) {
              score += inst->sibu_scores[t][r][r];
            }

            if (feat_opt.use_labeled_sibling) {
              int l = right->_label_s_t;
              score += inst->sibl_scores[t][r][r][l];
            }
          }

          const LatticeItem * const item = new LatticeItem(CMP,
                                                           t,
                                                           s,
                                                           score,
                                                           left,
                                                           right);

          lattice_insert(_lattice_cmp[t][s], item);
        }
      }
    }
  }
}

void Decoder2O::get_result(Instance * inst) {
  int len = inst->size();
  inst->predicted_heads.resize(len, -1);
  if (model_opt.labeled) {
    inst->predicted_deprelsidx.resize(len, -1);
  }

  const LatticeItem * best_item = _lattice_cmp[0][len - 1];
  __BUILD_TREE(inst, best_item);
}

void Decoder2O::free_lattice() {
  int len = _lattice_cmp.nrows();
  for (int i = 0; i < len; ++ i) {
    for (int j = 0; j < len; ++ j) {
      if (_lattice_incmp[i][j]) {
        delete _lattice_incmp[i][j];
      }

      if (_lattice_cmp[i][j]) {
        delete _lattice_cmp[i][j];
      }

      if (_lattice_sib[i][j]) {
        delete _lattice_sib[i][j];
      }
    }
  }
}


// ================================================================ //
// 2nd-order Decoder using dependency, sibling and grand features   //
// ================================================================ //
void Decoder2OCarreras::init_lattice(const Instance * inst) {
  int len = inst->size();
  _lattice_cmp.resize(len, len, len);
  _lattice_incmp.resize(len, len, L);

  _lattice_cmp = NULL;
  _lattice_incmp = NULL;

  for (int i = 0; i < len; ++ i) {
    _lattice_cmp[i][i][i] = new LatticeItem(i);
  }
}

void Decoder2OCarreras::decode_projective(const Instance * inst) {
  int len = inst->size();

  for (int width = 1; width < len; ++ width) {
    for (int s = 0; s + width < len; ++ s) {
      int t = s + width;

      // I(s, t) = C(s, r) + C(t, r + 1)
      for (int l = 0; l < L; ++ l) {
        for (int r = s; r < t; ++ r) {
          const LatticeItem * best_left_item = 0;
          double best_left_score = DOUBLE_NEG_INF;

          for (int cs = s; cs <= r; ++ cs) {
            if (cs == s && s != r) {
              continue;
            }

            const LatticeItem * item = _lattice_cmp[s][r][cs];

            if (!item) {
              continue;
            }

            double score = item->_prob;

            if (feat_opt.use_unlabeled_sibling) {
              score += inst->sibu_scores[s][t][cs];
            }

            if (feat_opt.use_labeled_sibling) {
              score += inst->sibl_scores[s][t][cs][l];
            }

            if (score > best_left_score) {
              best_left_item = item;
              best_left_score = score;
            }
          }

          const LatticeItem * best_right_item = 0;
          double best_right_score = DOUBLE_NEG_INF;

          for (int ct = r + 1; ct <= t; ++ ct) {
            if (ct == t &&  r + 1 != t) {
              continue;
            }

            const LatticeItem * item = _lattice_cmp[t][r + 1][ct];

            if (!item) {
              continue;
            }

            double score = item->_prob;

            if (feat_opt.use_unlabeled_grand &&
                (feat_opt.use_no_grand || ct != t)) {
              score += inst->grdu_scores[s][t][ct == t ? s : ct];
            }

            if (feat_opt.use_labeled_grand &&
                (feat_opt.use_no_grand || ct != t)) {
              score += inst->grdl_scores[s][t][ct == t ? s : ct][l];
            }

            if (score > best_right_score) {
              best_right_item = item;
              best_right_score = score;
            }
          }

          if (best_left_item && best_right_item) {
            double score = best_left_score + best_right_score;

            if (feat_opt.use_unlabeled_dependency) {
              score += inst->depu_scores[s][t];
            }

            if (feat_opt.use_labeled_dependency) {
              score += inst->depl_scores[s][t][l];
            }

            const LatticeItem * const item = new LatticeItem(INCMP,
                                                             s,
                                                             t,
                                                             score,
                                                             best_left_item,
                                                             best_right_item,
                                                             l);

            lattice_insert(_lattice_incmp[s][t][l], item);
          }   //  end for if !left || !right
        }
      }   //  end for for l = 0; l < L; ++ l

      if (s != 0) {
        // I(t, s) = C(s, r) + C(t, r + 1)
        for (int l = 0; l < L; ++ l) {
          for (int r = s; r < t; ++ r) {
            const LatticeItem * best_left_item = 0;
            double best_left_score = DOUBLE_NEG_INF;

            for (int cs = s; cs <= r; ++ cs) {
              if (cs == s && s != r) {
                continue;
              }

              const LatticeItem * item = _lattice_cmp[s][r][cs];

              if (!item) {
                continue;
              }

              double score = item->_prob;

              if (feat_opt.use_unlabeled_grand &&
                  (feat_opt.use_no_grand || cs != s)) {
                score += inst->grdu_scores[t][s][cs];
              }

              if (feat_opt.use_labeled_grand &&
                  (feat_opt.use_no_grand || cs != s)) {
                score += inst->grdl_scores[t][s][cs][l];
              }

              if (score > best_left_score) {
                best_left_item = item;
                best_left_score = score;
              }
            }

            const LatticeItem * best_right_item = 0;
            double best_right_score = DOUBLE_NEG_INF;

            for (int ct = r + 1; ct <= t; ++ ct) {
              if (ct == t && r + 1 != t) {
                continue;
              }

              const LatticeItem * item = _lattice_cmp[t][r + 1][ct];

              if (!item) {
                continue;
              }

              double score = item->_prob;

              if (feat_opt.use_unlabeled_sibling) {
                score += inst->sibu_scores[t][s][ct];
              }

              if (feat_opt.use_labeled_sibling) {
                score += inst->sibl_scores[t][s][ct][l];
              }

              if (score > best_right_score) {
                best_right_item = item;
                best_right_score = score;
              }
            }

            if (best_left_item && best_right_item) {
              double score = best_left_score + best_right_score;

              if (feat_opt.use_unlabeled_dependency) {
                score += inst->depu_scores[t][s];
              }

              if (feat_opt.use_labeled_dependency) {
                score += inst->depl_scores[t][s][l];
              }

              const LatticeItem * const item = new LatticeItem(INCMP,
                                                               t,
                                                               s,
                                                               score,
                                                               best_left_item,
                                                               best_right_item,
                                                               l);

              lattice_insert(_lattice_incmp[t][s][l], item);
            }
          }
        }
      }   //  end for if s != 0

      for (int m = s; m <= t; ++ m) {
        if (m != s) {   // C(s, t, m) = I(s, m, l) + C(m, t, cm);
          for (int l = 0; l < L; ++ l) {
            const LatticeItem * const left = _lattice_incmp[s][m][l];

            if (!left) {
              continue;
            }

            for (int cm = m; cm <= t; ++ cm) {
              if (cm == m && cm != t) {
                continue;
              }

              const LatticeItem * const right = _lattice_cmp[m][t][cm];

              if (!right) {
                continue;
              }

              double score = left->_prob + right->_prob;

              if (feat_opt.use_unlabeled_grand &&
                  (feat_opt.use_no_grand || cm != m)) {
                score += inst->grdu_scores[s][m][cm];
              }

              if (feat_opt.use_labeled_grand &&
                  (feat_opt.use_no_grand || cm != m)) {
                score += inst->grdl_scores[s][m][cm][l];
              }

              const LatticeItem * const item = new LatticeItem(CMP,
                                                               s,
                                                               t,
                                                               score,
                                                               left,
                                                               right);

              lattice_insert(_lattice_cmp[s][t][m], item);
            }   //  end for (int cm = m; cm <= t; ++ cm)
          }     //  enf for (int l = 0; l < L; ++ l)
        }       //  end for if (m != s)

        if (m != t && s != 0) { // C(t, s, m) = C(m, s, cm) + I(t, m, l)
          for (int l = 0; l < L; ++ l) {
            const LatticeItem * const right = _lattice_incmp[t][m][l];

            if (!right) {
              continue;
            }

            for (int cm = s; cm <= m; ++ cm) {
              if (cm == m && cm != s) {
                continue;
              }

              const LatticeItem * const left = _lattice_cmp[m][s][cm];

              if (!left) {
                continue;
              }

              double score = left->_prob + right->_prob;

              if (feat_opt.use_unlabeled_grand &&
                  (feat_opt.use_no_grand || cm != m)) {
                score += inst->grdu_scores[t][m][cm == m ? t : cm];
              }

              if (feat_opt.use_labeled_grand &&
                  (feat_opt.use_no_grand || cm != m)) {
                score += inst->grdl_scores[t][m][cm == m ? t : cm][l];
              }

              const LatticeItem * const item = new LatticeItem(CMP,
                                                               t,
                                                               s,
                                                               score,
                                                               left,
                                                               right);

              lattice_insert(_lattice_cmp[t][s][m], item);
            }   //  end for (int cm = s; cm <= m; ++ cm)
          }
        }
      }
    }
  }
}

void Decoder2OCarreras::get_result(Instance * inst) {
  int len = inst->size();
  inst->predicted_heads.resize(len, -1);
  if (model_opt.labeled) {
    inst->predicted_deprelsidx.resize(len, -1);
  }

  const LatticeItem * best_item = NULL;
  for (int c = 1; c < len; ++ c) {
    const LatticeItem * item = _lattice_cmp[0][len - 1][c];
    if (!item) {
      continue;
    }

    if (NULL == best_item || best_item->_prob < item->_prob) {
      best_item = item;
    }
  }

  __BUILD_TREE(inst, best_item);
}

void Decoder2OCarreras::free_lattice() {
  int len = _lattice_cmp.dim1();
  for (int i = 0; i < len; ++ i) {
    for (int j = 0; j < len; ++ j) {
      for (int l = 0; l < L; ++ l) {
        if (_lattice_incmp[i][j][l]) {
          delete _lattice_incmp[i][j][l];
        }
      }

      for (int k = 0; k < len; ++ k) {
        if (_lattice_cmp[i][j][k]) {
          delete _lattice_cmp[i][j][k];
        }
      }
    }
  }
}

}   //  end for namespace parser
}   //  end for namespace ltp
