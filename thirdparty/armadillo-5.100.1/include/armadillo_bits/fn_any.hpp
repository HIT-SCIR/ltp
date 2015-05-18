// Copyright (C) 2013-2015 Conrad Sanderson
// Copyright (C) 2013-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_any
//! @{



template<typename T1>
arma_inline
const mtOp<uword, T1, op_any>
any
  (
  const T1&   X,
  const uword dim = 0,
  const typename enable_if< is_arma_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return mtOp<uword, T1, op_any>(X, dim, 0);
  }



template<typename T1>
arma_inline
const mtOp<uword, T1, op_any>
any
  (
  const T1&   X,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtOp<uword, T1, op_any>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
bool
any
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return op_any::any_vec(X);
  }



template<typename T1>
inline
arma_warn_unused
bool
any(const mtOp<uword, T1, op_any>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("any(): two consecutive calls to any() detected");
  
  return op_any::any_vec(in.m);
  }



template<typename T1>
arma_inline
const Op< mtOp<uword, T1, op_any>, op_any>
any(const mtOp<uword, T1, op_any>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<uword, mtOp<uword, T1, op_any>, op_any>(in, dim, 0);
  }



//! @}
