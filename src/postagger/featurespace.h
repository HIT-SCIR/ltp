#ifndef __LTP_POSTAGGER_FEATURE_SPACE_H__
#define __LTP_POSTAGGER_FEATURE_SPACE_H__

#include <iostream>
#include <vector>

#include "utils/smartmap.hpp"

namespace ltp {
namespace postagger {

class FeatureSpaceIterator {
public:
  FeatureSpaceIterator()
    : _dicts(NULL),
      _i(0),
      _state(0) {
    // should be careful about the empty dicts
  }

  FeatureSpaceIterator(utility::SmartMap<int> * dicts,int num_dicts)
    : _dicts(dicts),
      _num_dicts(num_dicts),
      _i(0),
      _state(0) {
    ++ (*this);
  }

  ~FeatureSpaceIterator() {
  }

  const char * key() { return _j.key(); }
  int id() { return (*_j.value()); }
  int tid() { return _i; }

  bool operator ==(const FeatureSpaceIterator & other) const {
    return ((_dicts + _i) == other._dicts);
  }

  bool operator !=(const FeatureSpaceIterator & other) const {
    return ((_dicts + _i) != other._dicts);
  }

  FeatureSpaceIterator & operator = (const FeatureSpaceIterator & other) {
    if (this != &other) {
      _dicts  = other._dicts;
      _i    = other._i;
      _state  = other._state;
    }

    return *this;
  }

  void operator ++() {
    switch (_state) {
      case 0:
        for (_i = 0; _i < _num_dicts; ++ _i) {
          for (_j = _dicts[_i].begin(); _j != _dicts[_i].end(); ++ _j) {
            _state = 1;
            return;
      case 1:;
          }
        }
    }
  }

  int _i;
  int _state;
  int _num_dicts;
  utility::SmartMap<int>::const_iterator  _j;
  utility::SmartMap<int> * _dicts;
};

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
  int num_feature_groups();

  /*
   * dump the feature space to a output stream
   *
   *  @param[in]  ofs   the output stream
   */
  void dump(std::ostream & ofs);

  /*
   * load the feature space from a input stream
   *
   *  @param[in]  num_labels  the number of labels
   *  @param[in]  ifs     the input stream
   */
  bool load(int num_labeles, std::istream & ifs);

  FeatureSpaceIterator begin() {
    return FeatureSpaceIterator(dicts,_num_dicts);
  }

  FeatureSpaceIterator end() {
    return FeatureSpaceIterator(dicts + _num_dicts,_num_dicts);
  }

private:
  int _offset;
  int _num_dicts;
  int _num_labels;
  utility::SmartMap<int> * dicts;
};

}     //  end for namespace postagger
}     //  end for namespace ltp
#endif  //  end for __LTP_POSTAGGER_FEATURE_SPACE_H__
