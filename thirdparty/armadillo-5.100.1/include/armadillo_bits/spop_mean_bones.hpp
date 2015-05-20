// Copyright (C) 2012 Ryan Curtin
// Copyright (C) 2012 Conrad Sanderson
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_mean
//! @{


//! Class for finding mean values of a sparse matrix
class spop_mean
  {
  public:

  // Apply mean into an output sparse matrix (or vector).
  template<typename T1>
  inline static void apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_mean>& in);

  template<typename T1>
  inline static void apply_noalias(SpMat<typename T1::elem_type>& out, const SpProxy<T1>& p, const uword dim);
  
  // Take direct mean of a set of values.  Length of array and number of values can be different.
  template<typename eT>
  inline static eT direct_mean(const eT* const X, const uword length, const uword N);

  template<typename eT>
  inline static eT direct_mean_robust(const eT* const X, const uword length, const uword N);

  template<typename T1>
  inline static typename T1::elem_type mean_all(const SpBase<typename T1::elem_type, T1>& X);

  // Take the mean using an iterator.
  template<typename T1, typename eT>
  inline static eT iterator_mean(T1& it, const T1& end, const uword n_zero, const eT junk);

  template<typename T1, typename eT>
  inline static eT iterator_mean_robust(T1& it, const T1& end, const uword n_zero, const eT junk);
  };



//! @}
