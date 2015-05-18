// Copyright (C) 2011-2012 Conrad Sanderson
// Copyright (C) 2011-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_strans
//! @{



template<typename T1>
arma_inline
const Op<T1, op_strans>
strans
  (
  const T1& X,
  const typename enable_if< is_arma_type<T1>::value == true >::result* junk1 = 0,
  const typename arma_cx_only<typename T1::elem_type>::result*         junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_strans>(X);
  }



// NOTE: for non-complex objects, deliberately returning op_htrans instead of op_strans,
// NOTE: due to currently more optimisations available when using op_htrans, especially by glue_times
template<typename T1>
arma_inline
const Op<T1, op_htrans>
strans
  (
  const T1& X,
  const typename enable_if< is_arma_type<T1>::value == true >::result* junk1 = 0,
  const typename arma_not_cx<typename T1::elem_type>::result*          junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_htrans>(X);
  }



//! two consecutive transpose operations cancel each other
template<typename T1>
arma_inline
const T1&
strans(const Op<T1, op_strans>& X)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("strans(): removing op_strans");
  
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
strans(const T1& x)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1,spop_strans>(x);
  }



//! @}
