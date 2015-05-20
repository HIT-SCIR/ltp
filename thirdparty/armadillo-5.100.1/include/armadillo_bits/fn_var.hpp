// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_var
//! @{



template<typename T1>
inline
const mtOp<typename T1::pod_type, T1, op_var>
var
  (
  const T1& X,
  const uword norm_type = 0,
  const uword dim = 0,
  const typename enable_if< is_arma_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return mtOp<typename T1::pod_type, T1, op_var>(X, norm_type, dim);
  }



template<typename T1>
inline
const mtOp<typename T1::pod_type, T1, op_var>
var
  (
  const T1& X,
  const uword norm_type,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtOp<typename T1::pod_type, T1, op_var>(X, norm_type, dim);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::pod_type
var
  (
  const T1& X,
  const uword norm_type = 0,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return op_var::var_vec( X, norm_type );
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result
var(const T&)
  {
  return T(0);
  }



template<typename T1>
inline
const mtSpOp<typename T1::pod_type, T1, spop_var>
var
  (
  const T1& X,
  const uword norm_type = 0,
  const uword dim = 0,
  const typename enable_if< is_arma_sparse_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_sparse_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);

  return mtSpOp<typename T1::pod_type, T1, spop_var>(X, norm_type, dim);
  }



template<typename T1>
inline
const mtSpOp<typename T1::pod_type, T1, spop_var>
var
  (
  const T1& X,
  const uword norm_type,
  const uword dim = 0,
  const typename enable_if<resolves_to_sparse_vector<T1>::value == true>::result* junk1 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);

  return mtSpOp<typename T1::pod_type, T1, spop_var>(X, norm_type, dim);
  }



template<typename T1>
inline
typename T1::pod_type
var
  (
  const T1& X,
  const uword norm_type = 0,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_sparse_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);

  return spop_var::var_vec(X, norm_type);
  }



//! @}
