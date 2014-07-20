#ifndef __LTP_PARSER_DECODER_H__
#define __LTP_PARSER_DECODER_H__

#include "parser/instance.h"
#include "parser/settings.h"
#include "parser/options.h"
#include <iostream>
#include <list>

#include <stdlib.h>

namespace ltp {
namespace parser {

// data struct for decode chart item. Provide several construction
// methods and bind certain type.
class LatticeItem {
public:
  const int _g;           /* grand */
  const int _s;           /* from */
  const int _t;           /*< the distance to */
  const int _comp;        /*< specify if this span is complete */
  const int _label_s_t;   /*< label type */

  const LatticeItem * const _left;
  const LatticeItem * const _right;

  const double _prob;

public:
  LatticeItem(const int comp,
      const int g,
      const int s,
      const int t,
      const double prob,
      const LatticeItem * const left,
      const LatticeItem * const right) :
    _g(g),
    _s(s),
    _t(t),
    _comp(comp),
    _prob(prob),
    _left(left),
    _right(right),
    _label_s_t(-1) { }

  LatticeItem(const int comp,
      const int s,
      const int t,
      const double prob,
      const LatticeItem * const left,
      const LatticeItem * const right,
      const int label_s_t = -1) :
    _g(-1),
    _s(s),
    _t(t),
    _comp(comp),
    _prob(prob),
    _left(left),
    _right(right),
    _label_s_t(label_s_t) { }

  // for span like C(s,s)
  LatticeItem(const int g,
      const int s) :
    _g(g),
    _s(s),
    _t(s),
    _prob(0.0),
    _comp(CMP),
    _left(0),
    _right(0),
    _label_s_t(-1) { }

  LatticeItem(const int s) :
    _s(s),
    _t(s),
    _prob(0.0),
    _comp(CMP),
    _left(0),
    _right(0),
    _g(-1),
    _label_s_t(-1) { }

  ~LatticeItem() {}

private:
  // forbidden construction
  LatticeItem(const LatticeItem & rhs) :
    _s(0),
    _t(0),
    _prob(0.0),
    _comp(-1),
    _left(0),
    _right(0),
    _label_s_t(-1),
    _g(-1) {
    std::cerr << "LatticeItem::LatticeItem(const LatticeItem & rhs) is not allowed"
              << std::endl;
    exit(-1);
  }

  LatticeItem & operator = (const LatticeItem & rhs) {
    std::cerr << "LatticeItem::operator= (const LatticeItem & rhs) is not allowed"
              << std::endl;
    exit(-1);
  }

};

class Decoder {
public:
  Decoder() {}
  virtual ~Decoder() {}

  /*
   * Decode the instance, this method is a controller,
   * execute:
   *  - init lattice
   *  - decode projective
   *  - get result
   *  - free lattice
   * in sequence.
   *
   *  @param[in]  inst  the instance
   */
  void decode(Instance * inst) {
    init_lattice(inst);
    decode_projective(inst);
    get_result(inst);
    free_lattice();
  }

  virtual void init_lattice(const Instance * inst) = 0;
  virtual void decode_projective(const Instance * inst) = 0;
  virtual void get_result(Instance * inst) = 0;
  virtual void free_lattice() = 0;

protected:
  void lattice_insert(const LatticeItem * &position, const LatticeItem * const item) {
    if (position == NULL) {
      position = item;
    } else if (position->_prob < item->_prob - EPS) {
      delete position;
      position = item;
    } else {
      delete item;
    }
  }

  void __BUILD_TREE(Instance * inst, const LatticeItem * item) {
    if (!item) {
      return;
    }

    __BUILD_TREE(inst, item->_left);

    if (INCMP == item->_comp) {
      inst->predicted_heads[item->_t] = item->_s;

      if (model_opt.labeled) {
        inst->predicted_deprelsidx[item->_t] = item->_label_s_t;
      }
    } else if (CMP == item->_comp) {
      // do nothing;
    } else if (SIBSP == item->_comp) {
      // do nothing
    } else {
      // do nothing
    }

    __BUILD_TREE(inst, item->_right);
  }

};  //  end for class decoder

}   //  end for namespace parser
}   //  end for namespace ltp

#endif  // end for __LTP_PARSER_DECODER_H__
