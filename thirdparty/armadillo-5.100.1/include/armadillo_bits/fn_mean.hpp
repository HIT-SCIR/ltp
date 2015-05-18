// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_mean
//! @{



template<typename T1>
arma_inline
const Op<T1, op_mean>
mean
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
  
  return Op<T1, op_mean>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_mean>
mean
  (
  const T1& X,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_mean>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
mean
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return op_mean::mean_all(X);
  }



//! \brief
//! Immediate 'find mean value' operation,
//! invoked, for example, by: mean(mean(A))
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
mean(const Op<T1, op_mean>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("mean(): two consecutive mean() calls detected");
  
  return op_mean::mean_all(in.m);
  }



template<typename T1>
arma_inline
const Op< Op<T1, op_mean>, op_mean>
mean(const Op<T1, op_mean>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return Op< Op<T1, op_mean>, op_mean>(in, dim, 0);
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result &
mean(const T& x)
  {
  return x;
  }



template<typename T1>
inline
arma_warn_unused
const SpOp<T1, spop_mean>
mean
  (
  const T1& X,
  const uword dim = 0,
  const typename enable_if< is_arma_sparse_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_sparse_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();

  arma_ignore(junk1);
  arma_ignore(junk2);

  return SpOp<T1, spop_mean>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
const SpOp<T1, spop_mean>
mean
  (
  const T1& X,
  const uword dim,
  const typename enable_if< resolves_to_sparse_vector<T1>::value == true >::result* junk1 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);

  return SpOp<T1, spop_mean>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
mean
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if< resolves_to_sparse_vector<T1>::value == true >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();

  arma_ignore(junk1);
  arma_ignore(junk2);

  return spop_mean::mean_all(X);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
mean(const SpOp<T1, spop_mean>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("mean(): two consecutive mean() calls detected");

  return spop_mean::mean_all(in.m);
  }



//! @}
