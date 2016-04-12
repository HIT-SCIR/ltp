#ifndef __LTP_FRAMEWORK_FEATRUE_SPACE_H__
#define __LTP_FRAMEWORK_FEATRUE_SPACE_H__

#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"
#include "utils/smartmap.hpp"

namespace ltp {
namespace framework {

using boost::uint32_t;
using boost::int32_t;

class FeatureSpaceIterator {
public:
  FeatureSpaceIterator()
    : _dicts(NULL),
      _i(0),
      _state(0) {
    // should be careful about the empty dicts
  }

  FeatureSpaceIterator(const utility::SmartMap<int32_t>* dicts, uint32_t num_dicts)
    : _dicts(dicts),
      _num_dicts(num_dicts),
      _i(0),
      _state(0) {
    ++ (*this);
  }

  ~FeatureSpaceIterator() {
  }

  const char* key() { return _j.key(); }
  int32_t id() { return (*_j.value()); }
  uint32_t tid() { return _i; }

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

  uint32_t _i;
  uint32_t _num_dicts;
  uint32_t _state;
  const utility::SmartMap<int32_t>* _dicts;
  utility::SmartMap<int32_t>::const_iterator _j;
};

class ViterbiFeatureSpace {
public:
  ViterbiFeatureSpace(uint32_t nr_dicts, uint32_t nr_labels = 1)
    : _num_dicts(nr_dicts), _num_labels(nr_labels), _offset(0) {
    dicts = new utility::SmartMap<int32_t>[ nr_dicts ];
  }

  ~ViterbiFeatureSpace(void) {
    delete [](dicts);
  }

  int32_t retrieve(const uint32_t& tid, const char* key) const {
    int32_t val;
    if (dicts[tid].get(key, val)) {
      return val;
    }
    return -1;
  }

  int32_t retrieve(const uint32_t& tid, const std::string& key) const {
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
  int32_t retrieve(const uint32_t& tid, const char* key, bool create) {
    int32_t val;
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

  int32_t retrieve(const uint32_t& tid, const std::string& key, bool create) {
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
  int32_t index(const uint32_t& tid, const char* key, const uint32_t& lid = 0) const {
    int32_t idx = -1;
    if (!dicts[tid].get(key, idx)) {
      return -1;
    }
    return idx * _num_labels + lid;
  }

  int32_t index(const uint32_t& tid, const std::string& key, const uint32_t& lid = 0) const {
    return index(tid, key.c_str(), lid);
  }

  /**
   * return dimension of  the transform feature( \Phi(y1,y2) )
   *
   *  @param[in]  prev_lid   the previous label
   *  @param[in]  lid   the label
   *  @return     int   the dimension index
   */
  int32_t index(const uint32_t& prev_lid, const uint32_t& lid) const {
    return _offset * _num_labels + prev_lid * _num_labels + lid;
  }

  uint32_t num_features() const {
    return _offset;
  }

  uint32_t dim() const {
    return _offset* _num_labels + _num_labels* _num_labels;
  }

  uint32_t num_groups() const {
    return _offset + _num_labels;
  }

  uint32_t num_dicts() const {
    return _num_dicts;
  }

  void set_num_labels(const uint32_t& num_labels) {
    _num_labels = num_labels;
  }

  /**
   * dump the feature space to a output stream
   *
   *  @param[in]  ofs   the output stream
   */
  void dump(std::ostream & ofs) const {
    char chunk[16];
    uint32_t sz = _num_dicts;
    strncpy(chunk, "featurespace", 16);

    ofs.write(chunk, 16);
    ofs.write(reinterpret_cast<const char *>(&_offset), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char *>(&sz), sizeof(uint32_t));

    for (uint32_t i = 0; i < _num_dicts; ++ i) {
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
    uint32_t sz;
    ifs.read(chunk, 16);
    if (strcmp(chunk, "featurespace")) {
      return false;
    }

    ifs.read(reinterpret_cast<char *>(&_offset), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(&sz), sizeof(uint32_t));

    if (sz != _num_dicts) {
      return false;
    }

    for (uint32_t i = 0; i < sz; ++ i) {
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
  uint32_t _offset;
  uint32_t _num_labels;
  uint32_t _num_dicts;
  utility::SmartMap<int32_t>* dicts;
};

} //  namespace framework
} //  namespace ltp

#endif  //  end for __LTP_FRAMEWORK_FEATRUE_SPACE_H__
