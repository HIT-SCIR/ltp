#ifndef __LTP_SEGMENTOR_FEATURE_SPACE_H__
#define __LTP_SEGMENTOR_FEATURE_SPACE_H__

#include <iostream>
#include <vector>

#include "utils/smartmap.hpp"

namespace ltp {
namespace segmentor {

class FeatureSpaceIterator {
public:
  FeatureSpaceIterator()
    : _dicts(NULL),
      _i(0),
      _state(0) {
    // should be careful about the empty dicts
  }

  FeatureSpaceIterator(utility::SmartMap<int> * dicts, int num_dicts)
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
      _i      = other._i;
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

  /*
   * retrieve dimension of the feature
   *
   *  @param[in]  tid   the template index of the key
   *  @param[in]  key   the key value of the feature
   *  @param[in]  create   if create is ture, insert the key into the dict
   *  @return     int   the dimension index
   */
  int retrieve(int tid, const char * key, bool create);

  /*
   * return dimension of the key with certain label ( φ(x,y) )
   *
   *  @param[in]  tid   the template index of the key
   *  @param[in]  key   the key value
   *  @param[in]  lid   the label
   *  @return     int   the dimension index
   */
  int index(int tid, const char * key, int lid = 0) const;

  /*
   * return dimension of  the transform feature( φ(y1,y2) )
   *
   *  @param[in]  prev_lid   the previous label
   *  @param[in]  lid   the label
   *  @return     int   the dimension index
   */
  int index(int prev_lid, int lid) const;
  int num_features();
  int dim();
  int num_feature_groups();
  void set_num_labels(int num_labeles);

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
   *  @return     bool    true on success, otherwise false
   */
  bool load(int num_labeles, std::istream & ifs);

  FeatureSpaceIterator begin() {
    return FeatureSpaceIterator(dicts, _num_dicts);
  }

  FeatureSpaceIterator end() {
    return FeatureSpaceIterator(dicts + _num_dicts, _num_dicts);
  }
private:
  int _offset;
  int _num_labels;
  int _num_dicts;
  utility::SmartMap<int> * dicts;
};

}     //  end for namespace segmentor
}     //  end for namespace ltp
#endif  //  end for __LTP_SEGMENTOR_FEATURE_SPACE_H__
