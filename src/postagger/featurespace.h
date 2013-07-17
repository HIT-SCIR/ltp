#ifndef __LTP_POSTAGGER_FEATURE_SPACE_H__
#define __LTP_POSTAGGER_FEATURE_SPACE_H__

#include <iostream>
#include <vector>

#include "smartmap.hpp"

namespace ltp {
namespace postagger {

class FeatureSpace {
public:
    FeatureSpace(int num_labels = 1);
    ~FeatureSpace();

    int retrieve(int tid, const char * key, bool create);
    int index(int tid, const char * key, int lid = 0);
    int index(int prev_lid, int lid);
    int num_features();
    int dim();
    void set_num_labels(int num_labeles);

    /*
     * dump the feature space to a output stream
     *
     *  @param[in]  ofs     the output stream
     */
    void dump(std::ostream & ofs);

    /*
     * load the feature space from a input stream
     *
     *  @param[in]  num_labels  the number of labels
     *  @param[in]  ifs         the input stream
     */
    bool load(int num_labeles, std::istream & ifs);
private:
    int _offset;
    int _num_labels;
    std::vector< utility::SmartMap<int> * > dicts;
};

}       //  end for namespace postagger
}       //  end for namespace ltp
#endif  //  end for __LTP_POSTAGGER_FEATURE_SPACE_H__
