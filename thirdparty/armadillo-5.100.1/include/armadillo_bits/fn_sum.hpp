// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_sum
//! @{


//! \brief
//! Delayed sum of elements of a matrix along a specified dimension (either rows or columns).
//! The result is stored in a dense matrix that has either one column or one row.
//! For dim = 0, find the sum of each column (traverse across rows)
//! For dim = 1, find the sum of each row (traverse across columns)
//! The default is dim = 0.
//! NOTE: the dim argument is different than in Matlab/Octave.

template<typename T1>
arma_inline
const Op<T1, op_sum>
sum
  (
  const T1& X,
  const uword dim = 0,
  const typename enable_if< is_arma_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_sum>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_sum>
sum
  (
  const T1& X,
  const uword dim,
  const typename enable_if< resolves_to_vector<T1>::value == true >::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_sum>(X, dim, 0);
  }



//! \brief
//! Immediate 'sum all values' operation for expressions which resolve to a vector 
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
sum
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if< resolves_to_vector<T1>::value == true >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return accu(X);
  }



//! \brief
//! Immediate 'sum all values' operation,
//! invoked, for example, by: sum(sum(A))

template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
sum(const Op<T1, op_sum>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("sum(): two consecutive sum() calls detected");
  
  return accu(in.m);
  }



template<typename T1>
arma_inline
const Op<Op<T1, op_sum>, op_sum>
sum(const Op<T1, op_sum>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return Op<Op<T1, op_sum>, op_sum>(in, dim, 0);
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result &
sum(const T& x)
  {
  return x;
  }



//! sum of sparse object
template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value == true) && (resolves_to_sparse_vector<T1>::value == true),
  typename T1::elem_type
  >::result
sum(const T1& x)
  {
  arma_extra_debug_sigprint();
  
  // sum elements
  return accu(x);
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value == true) && (resolves_to_sparse_vector<T1>::value == false),
  const SpOp<T1,spop_sum>
  >::result
sum(const T1& x, const uword dim = 0)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1,spop_sum>(x, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
sum(const SpOp<T1, spop_sum>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("sum(): two consecutive sum() calls detected");
  
  return accu(in.m);
  }



template<typename T1>
arma_inline
const SpOp<SpOp<T1, spop_sum>, spop_sum>
sum(const SpOp<T1, spop_sum>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<SpOp<T1, spop_sum>, spop_sum>(in, dim, 0);
  }



//! @}
