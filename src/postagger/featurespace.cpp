#include "featurespace.h"

#include "extractor.h"

namespace ltp {
namespace postagger {

FeatureSpace::FeatureSpace(int num_labels) : 
    _num_labels(num_labels), 
    _offset(0) {
    dicts.resize(Extractor::num_templates());

    for (int i = 0; i < dicts.size(); ++ i) {
        dicts[i] = new utility::SmartMap<int>();
    }
}

FeatureSpace::~FeatureSpace(void) {
    for (int i = 0; i < dicts.size(); ++ i) {
        delete dicts[i];
    }
}

int FeatureSpace::retrieve(int tid, const char * key, bool create) {
    int val;

    if (dicts[tid]->get(key, val)) {
        return val;
    } else {
        if (create) {
            val = _offset;
            dicts[tid]->set(key, val);
            ++ _offset;

            return val;
        }
    }
 
    return -1;
}

int FeatureSpace::index(int tid, const char * key, int lid) {
    int idx = retrieve(tid, key, false);
    if (idx < 0) {
        return -1;
    }

    return idx * _num_labels + lid;
}

int FeatureSpace::index(int prev_lid, int lid) {
    return _offset * _num_labels + prev_lid * _num_labels + lid;
}

int FeatureSpace::num_features() {
    return _offset;
}

int FeatureSpace::dim() {
    return _offset * _num_labels + _num_labels * _num_labels;
}

void FeatureSpace::set_num_labels(int num_labels) {
    _num_labels = num_labels;
}
void FeatureSpace::dump(std::ostream & ofs) {
    char chunk[16];
    unsigned int sz = dicts.size();
    strncpy(chunk, "featurespace", 16);

    ofs.write(chunk, 16);
    ofs.write(reinterpret_cast<const char *>(&_offset), sizeof(int));
    ofs.write(reinterpret_cast<const char *>(&sz), sizeof(unsigned int));

    for (int i = 0; i < dicts.size(); ++ i) {
        dicts[i]->dump(ofs);
    }
}

bool FeatureSpace::load(int num_labels, std::istream & ifs) {
    _num_labels = num_labels;
    char chunk[16];
    unsigned int sz;
    ifs.read(chunk, 16);
    if (strcmp(chunk, "featurespace")) {
        return false;
    }

    ifs.read(reinterpret_cast<char *>(&_offset), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&sz), sizeof(unsigned int));

    if (sz != dicts.size()) {
        return false;
    }

    for (unsigned i = 0; i < sz; ++ i) {
        if (!dicts[i]->load(ifs)) {
            return false;
        }
    }

    return true;
}

}   //  end for namespace postagger
}   //  end for namespace ltp
