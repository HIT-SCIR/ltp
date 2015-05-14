#ifndef __LTP_FRAMEWORK_FEATRUE_SPACE_H__
#define __LTP_FRAMEWORK_FEATRUE_SPACE_H__

#include <iostream>
#include <vector>
#include "utils/smartmap.hpp"

namespace ltp {
namespace framework {

class FeatureSpaceIterator {
public:
  FeatureSpaceIterator()
    : _dicts(NULL),
      _i(0),
      _state(0) {
    // should be careful about the empty dicts
  }

  FeatureSpaceIterator(const utility::SmartMap<int>* dicts, int num_dicts)
    : _dicts(dicts),
      _num_dicts(num_dicts),
      _i(0),
      _state(0) {
    ++ (*this);
  }

  ~FeatureSpaceIterator() {
  }

  const char* key() { return _j.key(); }
  int id() { return (*_j.value()); }
  size_t tid() { return _i; }

  bool operator ==(const FeatureSpaceIterator & other) const {
    return ((_dicts + _i) == other._dicts);
  }

  bool operator !=(const FeatureSpaceIterator & other) const {
    return ((_dicts + _i) != other._dicts);
  }

  FeatureSpaceIterator& operator = (const FeatureSpaceIterator & other) {
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

  size_t _i;
  size_t _num_dicts;
  size_t _state;
  const utility::SmartMap<int>* _dicts;
  utility::SmartMap<int>::const_iterator _j;
};

class ViterbiFeatureSpace {
public:
  ViterbiFeatureSpace(size_t nr_dicts, size_t nr_labels = 1)
    : _num_dicts(nr_dicts), _num_labels(nr_labels), _offset(0) {
    dicts = new utility::SmartMap<int>[ nr_dicts ];
  }

  ~ViterbiFeatureSpace(void) {
    delete [](dicts);
  }

  int retrieve(const size_t& tid, const char* key) const {
    int val;
    if (dicts[tid].get(key, val)) {
      return val;
    }
    return -1;
  }

  int retrieve(const size_t& tid, const std::string& key) const {
    return retrieve(tid, key.c_str());
  }

  /**
   * retrieve dimension of the feature
   *
   *  @param[in]  tid   the template index of the key
   *  @param[in]  key   the key value of the feature
   *  @param[in]  create   if create is ture, insert the key into the dict
   *  @return     int   the dimension index
   */
  int retrieve(const size_t& tid, const char* key, bool create) {
    int val;
    if (dicts[tid].get(key, val)) {
      return val;
    } else {
      if (create) {
        val = _offset;
        dicts[tid].set(key, val);
        ++ _offset;
        return val;
      }
    }
    return -1;
  }

  int retrieve(const size_t& tid, const std::string& key, bool create) {
    return retrieve(tid, key.c_str(), create);
  }

  /**
   * return dimension of the key with certain label ( \Phi(x,y) )
   *
   *  @param[in]  tid   the template index of the key
   *  @param[in]  key   the key value
   *  @param[in]  lid   the label
   *  @return     int   the dimension index
   */
  int index(const size_t& tid, const char* key, const size_t& lid = 0) const {
    int idx = -1;
    if (!dicts[tid].get(key, idx)) {
      return -1;
    }
    return idx * _num_labels + lid;
  }

  int index(const size_t& tid, const std::string& key, const size_t& lid = 0) const {
    return index(tid, key.c_str(), lid);
  }

  /**
   * return dimension of  the transform feature( \Phi(y1,y2) )
   *
   *  @param[in]  prev_lid   the previous label
   *  @param[in]  lid   the label
   *  @return     int   the dimension index
   */
  int index(const size_t& prev_lid, const size_t& lid) const {
    return _offset * _num_labels + prev_lid * _num_labels + lid;
  }

  size_t num_features() const {
    return _offset;
  }

  size_t dim() const {
    return _offset* _num_labels + _num_labels* _num_labels;
  }

  size_t num_groups() const {
    return _offset + _num_labels;
  }

  size_t num_dicts() const {
    return _num_dicts;
  }

  void set_num_labels(const size_t& num_labels) {
    _num_labels = num_labels;
  }

  /**
   * dump the feature space to a output stream
   *
   *  @param[in]  ofs   the output stream
   */
  void dump(std::ostream & ofs) {
    char chunk[16];
    size_t sz = _num_dicts;
    strncpy(chunk, "featurespace", 16);

    ofs.write(chunk, 16);
    ofs.write(reinterpret_cast<const char *>(&_offset), sizeof(size_t));
    ofs.write(reinterpret_cast<const char *>(&sz), sizeof(size_t));

    for (size_t i = 0; i < _num_dicts; ++ i) {
      dicts[i].dump(ofs);
    }
  }

  /**
   * load the feature space from a input stream
   *
   *  @param[in]  num_labels  the number of labels
   *  @param[in]  ifs         the input stream
   *  @return     bool        true on success, otherwise false
   */
  bool load(std::istream& ifs) {
    char chunk[16];
    size_t sz;
    ifs.read(chunk, 16);
    if (strcmp(chunk, "featurespace")) {
      return false;
    }

    ifs.read(reinterpret_cast<char *>(&_offset), sizeof(size_t));
    ifs.read(reinterpret_cast<char *>(&sz), sizeof(size_t));

    if (sz != _num_dicts) {
      return false;
    }

    for (size_t i = 0; i < sz; ++ i) {
      if (!dicts[i].load(ifs)) {
        return false;
      }
    }
    return true;
  }

  FeatureSpaceIterator begin() const {
    return FeatureSpaceIterator(dicts, _num_dicts);
  }

  FeatureSpaceIterator end() const {
    return FeatureSpaceIterator(dicts + _num_dicts, _num_dicts);
  }
private:
  size_t _offset;
  size_t _num_labels;
  size_t _num_dicts;
  utility::SmartMap<int>* dicts;
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_FEATRUE_SPACE_H__
