// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_max
//! @{


//! \brief
//! Delayed 'maximum values' operation.
//! The dimension, along which the maxima are found, is set via 'dim'.
//! For dim = 0, the maximum value of each column is found (i.e. searches by traversing across rows).
//! For dim = 1, the maximum value of each row is found (i.e. searches by traversing across columns).
//! The default is dim = 0.

template<typename T1>
arma_inline
const Op<T1, op_max>
max
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
  
  return Op<T1, op_max>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_max>
max
  (
  const T1& X,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_max>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
max
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return op_max::max(X);
  }



//! \brief
//! Immediate 'find maximum value' operation,
//! invoked, for example, by: max(max(A))
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
max(const Op<T1, op_max>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("max(): two consecutive max() calls detected");
  
  return op_max::max(in.m);
  }



template<typename T1>
arma_inline
const Op< Op<T1, op_max>, op_max>
max(const Op<T1, op_max>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return Op< Op<T1, op_max>, op_max>(in, dim, 0);
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result &
max(const T& x)
  {
  return x;
  }



//! element-wise maximum
template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  ( is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value ),
  const Glue<T1, T2, glue_max>
  >::result
max
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return Glue<T1, T2, glue_max>(X, Y);
  }



template<typename T1>
inline
arma_warn_unused
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value == true) && (resolves_to_sparse_vector<T1>::value == true),
  typename T1::elem_type
  >::result
max(const T1& x)
  {
  arma_extra_debug_sigprint();
  
  return spop_max::vector_max(x);
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value == true) && (resolves_to_sparse_vector<T1>::value == false),
  const SpOp<T1, spop_max>
  >::result
max(const T1& X, const uword dim = 0)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1, spop_max>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
max(const SpOp<T1, spop_max>& X)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("max(): two consecutive max() calls detected");
  
  return spop_max::vector_max(X.m);
  }



template<typename T1>
inline
const SpOp< SpOp<T1, spop_max>, spop_max>
max(const SpOp<T1, spop_max>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return SpOp< SpOp<T1, spop_max>, spop_max>(in, dim, 0);
  }



//! @}
