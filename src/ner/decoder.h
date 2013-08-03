#ifndef __LTP_NER_DECODER_H__
#define __LTP_NER_DECODER_H__

#include <iostream>
#include <vector>
#include "instance.h"
#include "mat.h"
#include "rulebase.h"

namespace ltp {
namespace ner {

// data structure for lattice item
class LatticeItem {
public:
    LatticeItem (int _i, int _l, double _score, const LatticeItem * _prev) : 
        i(_i),
        l(_l),
        score(_score),
        prev(_prev) {}

    LatticeItem (int _l, double _score) : 
        i(0),
        l(_l),
        score(_score),
        prev(0) {}

public:
    int                 i;
    int                 l;
    double              score;
    const LatticeItem * prev;
};

class Decoder {
public:
    Decoder (int _L, rulebase::RuleBase & _base) : L(_L), base(_base) {}
    void decode(Instance * inst);

private:
    void init_lattice(const Instance * inst);
    void viterbi_decode(const Instance * inst);
    void get_result(Instance * inst);
    void free_lattice();

private:
    int L;

    math::Mat< const LatticeItem * > lattice;
    rulebase::RuleBase base;

    void lattice_insert(const LatticeItem * &position, const LatticeItem * const item) {
        if (position == NULL) {
            position = item;
        } else if (position->score < item->score) {
            delete position;
            position = item;
        } else {
            delete item;
        }
    }
};

// maintain kest best list of 
/*class KBestDecoder {
public:
    typedef std::vector< std::vector<int> > KBestDecodeResult;

public:
    KBestDecoder (int _L) : L(_L) {}

    void decode(Instance * inst, KBestDecodeResult & result);
private:
    void init_lattice(const Instance * inst);
    void kbest_viterbi_decode(const Instance * inst);
    void get_result(KBestDecodeResult & result);
    void free_lattice();

private:
    int L;

    Mat< KHeap< const LatticeItem * > > lattice;
};*/

}           //  end for namespace ner
}           //  end for namespace ltp
#endif      //  end for __LTP_NER_DECODER_H__
