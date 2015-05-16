#ifndef __SPARSE_VECTOR_H__
#define __SPARSE_VECTOR_H__

#include "featurevec.h"
#include "utils/unordered_map.hpp"
#include <iostream>
#include <string>

namespace ltp {
namespace math {

class SparseVec {
public:

  typedef std::unordered_map<int,double> internal_sparsevec_t;

  typedef internal_sparsevec_t::iterator     iterator;
  typedef internal_sparsevec_t::const_iterator const_iterator;

  SparseVec() {}
  ~SparseVec() {}

  const_iterator begin() const {
    return _vec.begin();
  }

  const_iterator end() const {
    return _vec.end();
  }

  iterator mbegin() {
    return _vec.begin();
  }

  iterator mend() {
    return _vec.end();
  }

  inline int dim() const {
    return _vec.size();
  }

  inline double L2() const {
    double norm = 0;
    for (const_iterator itx = _vec.begin();
        itx != _vec.end(); ++ itx) {
      double val = itx->second;
      norm += (val * val);
    }
    return norm;
  }

  inline void add(int idx,
                  double scale) {
    if (_vec.find(idx) == _vec.end()) _vec[idx] = 0.;
    _vec[idx] += scale;
  }

  inline void add(const SparseVec &other,
                  const double scale) {
    for (const_iterator itx = other.begin();
        itx != other.end(); ++ itx) {
      int idx = itx->first;
      if (_vec.find(idx) == _vec.end()) _vec[idx] = 0.;
      _vec[idx] += (scale * itx->second);
    }
  }

  void update_counter(int * updates,
                      int offset,
                      int num_labels) {

    int tmp = offset*num_labels;
    for (const_iterator itx = this->begin();
        itx != this->end(); ++ itx) {
      int idx = itx->first;
      if(idx < tmp) {//this means unfeatrues
        if(itx->second!=0.0){
          //std::cout<<"idx:"<<idx<<" value:"<<itx->second<<" +1"<<std::endl;
          updates[idx/num_labels]++;
        }
      }
    }
  }

  inline void add(const int * idx,
      const double * val,
      const int n,
      const double scale) {
    if (!idx) {
      return;
    }
    // int n = other->n;
    // const int * idx = other->idx;
    // const double * val = other->val;

    if (val == NULL) {
      for (int i = 0; i < n; ++ i) {
        if (_vec.find(idx[i]) == _vec.end()) _vec[idx[i]] = 0.;
        _vec[idx[i]] += scale;
      }
    } else {
      for (int i = 0; i < n; ++ i) {
        _vec[idx[i]] += (scale * val[i]);
      }
    }
  }

  inline void add(const int * idx,
      const double * val,
      const int n,
      const int loff,
      const double scale) {
    if (!idx) {
      return ;
    }

    if (val == NULL) {
      for (int i = 0; i < n; ++ i) {
        int id = idx[i] + loff;
        if (_vec.find(id) == _vec.end()) _vec[id] = 0.;
        _vec[id] += scale;
      }
    } else {
      for (int i = 0; i < n; ++ i) {
        int id = idx[i] + loff;
        if (_vec.find(id) == _vec.end()) _vec[id] = 0.;
        _vec[id] += (scale * val[i]);
      }
    }
  }

  inline void zero() {
    _vec.clear();
  }

  inline void str(std::ostream & out, std::string prefix = "  ") const {
    int i = 0;
    out << "{ ";
    for (const_iterator itx = _vec.begin();
        itx != _vec.end();
        ++ itx) {
      out << itx->first << ":" << itx->second << ", ";
      ++ i;

      if (i % 10 == 0) {
        out << "\n" << prefix;
      }
    }
    out << "}" << std::endl;
  }
private:
  internal_sparsevec_t _vec;
  double norm;
};

}     //  end for namespace math
}     //  end for namespace ltp

#endif  //  end for __SPARSE_VECTOR_H__
