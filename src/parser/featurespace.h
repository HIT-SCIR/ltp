#ifndef __LTP_PARSER_FEATURE_SPACE_H__
#define __LTP_PARSER_FEATURE_SPACE_H__

#include <iostream>
#include <vector>

#include "parser/instance.h"
#include "parser/collections.h"
#include "utils/smartmap.hpp"

namespace ltp {
namespace parser {
class FeatureSpaceIterator {
public:
  FeatureSpaceIterator()
    : _dicts(NULL),
      _i(0),
      _state(0){
  }

  FeatureSpaceIterator(DictionaryCollections *dicts)
    : _dicts(dicts),
      _i(0),
      _state(0) {
    ++(*this);
  }

  ~FeatureSpaceIterator(){
  }

  const char * key() { return _j.key(); }
  int id() { return (*_j.value()); }
  int tid() { return _i; }

  bool end() {
    int x = _dicts->size();
    if((x) == _i) {
      return true;
    }
    return false;
  }

  FeatureSpaceIterator & operator =(const FeatureSpaceIterator & other) {
    _dicts  = other._dicts;
    _i    = other._i;
    _state  = other._state;

    return *this;
  }

  void operator ++() {
    switch (_state) {
      case 0:
        for (_i = 0; ; ++_i) {
          if(NULL == _dicts->getDictionary(_i)) {
            return;
          }

          if (_dicts->getDictionary(_i)->database.begin() ==
              _dicts->getDictionary(_i)->database.end()) {
            _state = 1;
            return;
          }

          for (_j = _dicts->getDictionary(_i)->database.begin();
              _j != _dicts->getDictionary(_i)->database.end();
              ++_j) {
            _state = 1;
            return;
      case 1:;
          }
        }
    }
  }

  int getI() {
    return _i;
  }

  DictionaryCollections * getDicts() {
    return _dicts;
  }

private:
  int _i;
  int _state;
  utility::SmartMap<int>::const_iterator _j;
  DictionaryCollections * _dicts;
};

/*
 * Feature Space Class, used to process features
 *
 */
class FeatureSpace {
public:
  enum { DEP, SIB, GRD, GRDSIB, POSU, POSB };

public:
  /*
   * Constructor of feature space, construct feature space with
   * instances. Number of dependency relations is given when
   * constructing the feature space.
   *
   *  @param[in]  _num_deprels  the number of dependency relations
   */
  FeatureSpace() : _num_deprels(1), _offset(0) {
    // set all the group to be empty
    for (int i = 0; i < NUM_FEATURE_GROUPS; ++ i) {
      groups[i]  = 0;
      offsets[i] = 0;
    }
  }

  ~FeatureSpace() {
    for (int i = 0; i < NUM_FEATURE_GROUPS; ++ i) {
      if (groups[i]) {
        delete groups[i];
      }
    }
  }

  /*
   * Build feature space from the instances. For dependency parsing,
   * there is no negative feature, so the feature space can be maintained
   * once the training instance is given.
   *
   *  @param[in]  num_deprels
   *  @param[in]  instances     the instances
   */
  int build_feature_space(int num_deprels, const std::vector<Instance *> & instances);

  /*
   * Build feature space for truncate, just like the function above
   *
   *  @param
   */
  void build_feature_space_truncate(int num_deprels);

  /*After copy the dic item from the model that not equal to Zero,set the offset */
  void set_offset_truncate();

  /*
   * Retrieve the feature string and maintain its index in the
   * feature space.
   *
   *  @param[in]  gid,  the group index
   *  @parma[in]  tid,  the template index
   *  @param[in]  key,  the key
   *  @param[in]  create, if true, a new key will be created.
   *  @return   int   index of the key, -1 if not exist.
   */
  int retrieve(int gid, int tid, const char * key, bool create);


  /*
   * Maintain the index of this key with certain label, for the
   * labeled case, lid is automically set as zero.
   *
   *  @param[in]  gid,  the group index
   *  @param[in]  tid,  the template index
   *  @param[in]  key,  the key
   *  @param[in]  lid,  the label index, default as zero
   *  @return   int   the index of the (key, label)
   */
  int index(int gid, int tid, const char * key, int lid = 0);

  /*
   * Get number of dimension of the features. For example, supposing there
   * are two groups of features whose number of feature is 10 and 20. and 6
   * is the number of dependency relations, then 10 * 6 + 20 * 6 is return.
   *
   *  @return   int   the number of labels
   */
  int dim();

  /*
   * Get number of dimension of the features. For example, supposing there
   * are two groups of features whose number of feature is 10 and 20, then
   * 10 + 20 is return.
   *
   *  @return   int   the number of labels
   */
  int num_features();

  /*
   * save the features space to the output stream
   *
   *  @param[out] out   the output stream
   */
  void save(ostream & out);

  /*
   * load feature space from input stream
   *
   *  @param[in]  num_deprels the number of dependency relation type
   *  @param[in]  in      the input stream
   */
  bool load(int num_deprels, istream & in);

  FeatureSpaceIterator begin(int gid) {
    return FeatureSpaceIterator(groups[gid]);
  }
private:
  const static int NUM_FEATURE_GROUPS = 6;

  int _offset;
  int _num_features;
  int _num_deprels;

  int offsets[NUM_FEATURE_GROUPS]; /*< offset of group */
  DictionaryCollections * groups[NUM_FEATURE_GROUPS];

private:
  /*
   * Allocate the feature groups according the options
   *
   *  @return   int     number of actived dictinary group
   */
  int allocate_dictionary_groups();
};

}   // end for namespace parser
}   // end for namespace ltp

#endif  //  end for __LTP_PARSER_FEATURE_SPACE_H__

