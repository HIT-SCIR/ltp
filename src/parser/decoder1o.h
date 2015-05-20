#ifndef __LTP_PARSER_DECODER_1_O_H__
#define __LTP_PARSER_DECODER_1_O_H__

#include "parser/instance.h"
#include "parser/decoder.h"
#include "parser/options.h"
#include "parser/debug.h"
#include "utils/math/mat.h"

namespace ltp {
namespace parser {

using namespace ltp::math;

class Decoder1O : public Decoder {
public:
  Decoder1O(int _l = 1) : L(_l) {}

protected:
  void init_lattice(const Instance * inst);
  void decode_projective(const Instance * inst);
  void get_result(Instance * inst);
  void free_lattice();
protected:
  int L;

  Mat< const LatticeItem * >  _lattice_cmp;   //  complete span
  Mat3< const LatticeItem * >   _lattice_incmp; //  incomplete span
};    //  end for class Decoder1O

}     //  end for namespace parser
}     //  end for namespace ltp

#endif  //  end for __LTP_PARSER_DECODER_1_O_H__
