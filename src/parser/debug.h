#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cassert>
#include "parser/instance.h"

namespace ltp {
namespace parser {

inline void instance_verify(const Instance * inst, ostream & out, bool show_features) {
    out << "instance : {" << endl;
    out << "  basic: {" << endl;
    for (int i = 0; i < inst->size(); ++ i) {
        out << "    [" << i << "]\t" 
            << inst->forms[i] 
            << "\t"
            << inst->postags[i]
            << "\t"
            << inst->heads[i];
        if (inst->predicted_heads.size() > 0) {
            out << "("
                << inst->predicted_heads[i]
                << ")";
        }

        if (inst->deprels.size() > 0) {
            out << "\t"
                << inst->deprels[i];
            if (inst->deprelsidx.size() > 0) {
                out << "["
                    << inst->deprelsidx[i]
                    << "]";
            }
 
            if (inst->predicted_deprelsidx.size() > 0) {
                out << "-["
                    << inst->predicted_deprelsidx[i]
                    << "]";
            }
        }

        out << endl;
    }
    out << "  }," << endl;
    if (show_features) {
        out << "  gold-features: ";
        inst->features.str(out, "    ");

        out << "  predicted-features: ";
        inst->predicted_features.str(out, "    ");
    }

    out << "}" << endl;
}

inline Instance * instance_example() {
    Instance * inst = new Instance;
    inst->forms.push_back("ROOT");  inst->postags.push_back("#RT"); inst->heads.push_back(-1);
    inst->forms.push_back("We");    inst->postags.push_back("NN");  inst->heads.push_back(1);
    inst->forms.push_back("like");  inst->postags.push_back("VV");  inst->heads.push_back(0);
    inst->forms.push_back("cat");   inst->postags.push_back("NN");  inst->heads.push_back(1);
    return inst;
}

}       //  end for namespace parser
}       //  end for namespace ltp

#endif  //  end for __DEBUG_H__
