// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_inv
//! @{



//! delayed matrix inverse (general matrices)
template<typename T1>
arma_inline
const Op<T1, op_inv>
inv
  (
  const Base<typename T1::elem_type,T1>& X,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_inv>(X.get_ref(), ((slow == false) ? 0 : 1), 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_inv>
inv
  (
  const Base<typename T1::elem_type,T1>& X,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "inv(): unknown method specified" );
  
  return Op<T1, op_inv>(X.get_ref(), ((sig == 'f') ? 0 : 1), 0);
  }



//! delayed matrix inverse (triangular matrices)
template<typename T1>
arma_inline
const Op<T1, op_inv_tr>
inv
  (
  const Op<T1, op_trimat>& X,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(slow);
  arma_ignore(junk);
  
  return Op<T1, op_inv_tr>(X.m, X.aux_uword_a, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_inv_tr>
inv
  (
  const Op<T1, op_trimat>& X,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "inv(): unknown method specified" );
  
  return Op<T1, op_inv_tr>(X.m, X.aux_uword_a, 0);
  }



template<typename T1>
inline
bool
inv
  (
         Mat<typename T1::elem_type>&    out,
  const Base<typename T1::elem_type,T1>& X,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  try
    {
    out = inv(X,slow);
    }
  catch(std::runtime_error&)
    {
    return false;
    }
  
  return true;
  }



template<typename T1>
inline
bool
inv
  (
         Mat<typename T1::elem_type>&    out,
  const Base<typename T1::elem_type,T1>& X,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  try
    {
    out = inv(X,method);
    }
  catch(std::runtime_error&)
    {
    return false;
    }
  
  return true;
  }



//! inverse of symmetric positive definite matrices
template<typename T1>
arma_inline
const Op<T1, op_inv_sympd>
inv_sympd
  (
  const Base<typename T1::elem_type, T1>& X,
  const char* method = "std",
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "inv_sympd(): unknown method specified" );
  
  return Op<T1, op_inv_sympd>(X.get_ref(), 0, 0);
  }



template<typename T1>
inline
bool
inv_sympd
  (
         Mat<typename T1::elem_type>&    out,
  const Base<typename T1::elem_type,T1>& X,
  const char* method = "std",
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  try
    {
    out = inv_sympd(X,method);
    }
  catch(std::runtime_error&)
    {
    return false;
    }
  
  return true;
  }



//! @}
