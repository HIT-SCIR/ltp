// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_trans
//! @{


template<typename T1>
arma_inline
const Op<T1, op_htrans>
trans
  (
  const T1& X,
  const typename enable_if< is_arma_type<T1>::value == true >::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_htrans>(X);
  }



template<typename T1>
arma_inline
const Op<T1, op_htrans>
htrans
  (
  const T1& X,
  const typename enable_if< is_arma_type<T1>::value == true >::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_htrans>(X);
  }



//! two consecutive transpose operations cancel each other
template<typename T1>
arma_inline
const T1&
trans(const Op<T1, op_htrans>& X)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("trans(): removing op_htrans");
  
  return X.m;
  }



template<typename T1>
arma_inline
const T1&
htrans(const Op<T1, op_htrans>& X)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("htrans(): removing op_htrans");
  
  return X.m;
  }



//
// handling of sparse matrices


template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value,
  const SpOp<T1,spop_strans>
  >::result
trans
  (
  const T1& x,
  const typename arma_not_cx<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return SpOp<T1,spop_strans>(x);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value,
  const SpOp<T1,spop_htrans>
  >::result
trans
  (
  const T1& x,
  const typename arma_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return SpOp<T1,spop_htrans>(x);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value,
  const SpOp<T1,spop_strans>
  >::result
htrans
  (
  const T1& x,
  const typename arma_not_cx<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return SpOp<T1,spop_strans>(x);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value,
  const SpOp<T1,spop_htrans>
  >::result
htrans
  (
  const T1& x,
  const typename arma_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return SpOp<T1,spop_htrans>(x);
  }



//! @}
