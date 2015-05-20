#ifndef __LTP_PARSER_DECODER_2_O_H__
#define __LTP_PARSER_DECODER_2_O_H__

#include "parser/decoder.h"

namespace ltp {
namespace parser {

// 2nd-order decoder with dependency features and sibling features
class Decoder2O : public Decoder {
public:
  Decoder2O(int _l = 1) : L(_l) {}

public:
  void init_lattice(const Instance * inst);
  void decode_projective(const Instance * inst);
  void get_result(Instance * inst);
  void free_lattice();
private:
  int L;
  Mat< const LatticeItem * > _lattice_cmp;
  Mat< const LatticeItem * > _lattice_incmp;
  Mat< const LatticeItem * > _lattice_sib;

};

// 2nd-order decoder with dependency, sibling and grand features
class Decoder2OCarreras : public Decoder {
public:
  Decoder2OCarreras(int _l = 1) : L(_l) {}

public:
  void init_lattice(const Instance * inst);
  void decode_projective(const Instance * inst);
  void get_result(Instance *  inst);
  void free_lattice();
private:
  int L;
  Mat3< const LatticeItem * > _lattice_cmp;
  Mat3< const LatticeItem * > _lattice_incmp;

};

}   //  end for namespace parser
}   //  end for namespace ltp

#endif  //  end for __LTP_PARSER_DECODER_2_O_H__
