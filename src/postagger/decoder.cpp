#include "decoder.h"

namespace ltp {
namespace postagger {


void
Decoder::decode(Instance * inst,const Poslexicon* lexicon ) {
  init_lattice(inst);
  viterbi_decode(inst,lexicon);
  get_result(inst);
  free_lattice();
}

void
Decoder::init_lattice(const Instance * inst) {
  int len = inst->size();
  lattice.resize(len, L);
  lattice = NULL;
}

void Decoder::viterbi_decode_inner(const Instance * inst,int i,int l){
  if (i == 0) {
    LatticeItem * item = new LatticeItem(i, l, inst->uni_scores[i][l], NULL);
    lattice_insert(lattice[i][l], item);
  } else {
    for (int pl = 0; pl < L; ++ pl) {
      double score = 0.;
      const LatticeItem * prev = lattice[i-1][pl];

      if (!prev) {
        continue;
      }

      score = inst->uni_scores[i][l] + inst->bi_scores[pl][l] + prev->score;
      const LatticeItem * item = new LatticeItem(i, l, score, prev);
      lattice_insert(lattice[i][l], item);
    }
  }   //  end for if i != 0
}

void
Decoder::viterbi_decode(const Instance * inst,const Poslexicon*  lexicon) {
  int len = inst->size();
  if (!lexicon){
    for (int i = 0; i < len; ++ i) {
      for (int l = 0; l < L; ++ l) {
        viterbi_decode_inner(inst,i,l);
      } // end for label kinds
    } // end for len
  } // end for lexicon == NULL
  else{
    std::vector <int> lex_labels;
    int lex_labels_size;
    for (int i = 0; i < len; ++ i) {
      if ( lexicon->get(inst->forms[i] ,lex_labels) ){
        lex_labels_size = lex_labels.size();
        for (int l_idx = 0; l_idx < lex_labels_size; ++l_idx) {
          viterbi_decode_inner( inst,i,lex_labels[l_idx] );
        } // end for label kinds
      }// end for if word found in lexicon
      else{
        for (int l = 0; l < L; ++ l) {
          viterbi_decode_inner(inst,i,l);
        } // end for label kinds
      }// end for word not found in lexicon
    } // end for len
  } // end for lexicon != NULL
}

void
Decoder::get_result(Instance * inst) {
  int len = inst->size();
  const LatticeItem * best_item = NULL;
  for (int l = 0; l < L; ++ l) {
    if (!lattice[len-1][l]) {
      continue;
    }
    if (best_item == NULL || lattice[len - 1][l]->score > best_item->score) {
      best_item = lattice[len - 1][l];
    }
  }

  const LatticeItem * item = best_item;
  inst->predicted_tagsidx.resize(len);

  while (item) {
    inst->predicted_tagsidx[item->i] = item->l;
    // std::cout << item->i << " " << item->l << std::endl;
    item = item->prev;
  }
}

void
Decoder::free_lattice() {
  int len = lattice.total_size();
  const LatticeItem ** p = lattice.c_buf();
  for (int i = 0; i < len; ++ i) {
    if (p[i]) {
      delete p[i];
    }
  }
}

}     //  end for namespace postagger
}     //  end for namespace ltp

