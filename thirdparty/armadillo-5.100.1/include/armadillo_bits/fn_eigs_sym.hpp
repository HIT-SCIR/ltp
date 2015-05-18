// Copyright (C) 2013-2014 Ryan Curtin
// Copyright (C) 2013-2014 Conrad Sanderson
// Copyright (C) 2013-2014 NICTA
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_eigs_sym
//! @{


//! eigenvalues of symmetric real sparse matrix X
template<typename T1>
inline
Col<typename T1::pod_type>
eigs_sym
  (
  const SpBase<typename T1::elem_type,T1>& X,
  const uword                              n_eigvals,
  const char*                              form = "lm",
  const typename T1::elem_type             tol  = 0.0,
  const typename arma_real_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  Mat<typename T1::elem_type> eigvec;
  Col<typename T1::pod_type > eigval;
  
  const bool status = sp_auxlib::eigs_sym(eigval, eigvec, X, n_eigvals, form, tol);
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eigs_sym(): failed to converge");
    }
  
  return eigval;
  }



//! eigenvalues of symmetric real sparse matrix X
template<typename T1>
inline
bool
eigs_sym
  (
           Col<typename T1::pod_type >&    eigval,
  const SpBase<typename T1::elem_type,T1>& X,
  const uword                              n_eigvals,
  const char*                              form = "lm",
  const typename T1::elem_type             tol  = 0.0,
  const typename arma_real_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  Mat<typename T1::elem_type> eigvec;
  
  const bool status = sp_auxlib::eigs_sym(eigval, eigvec, X, n_eigvals, form, tol);
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eigs_sym(): failed to converge", false);
    }
  
  return status;
  }



//! eigenvalues and eigenvectors of symmetric real sparse matrix X
template<typename T1>
inline
bool
eigs_sym
  (
           Col<typename T1::pod_type >&    eigval,
           Mat<typename T1::elem_type>&    eigvec,
  const SpBase<typename T1::elem_type,T1>& X,
  const uword                              n_eigvals,
  const char*                              form = "lm",
  const typename T1::elem_type             tol  = 0.0,
  const typename arma_real_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  arma_debug_check( void_ptr(&eigval) == void_ptr(&eigvec), "eigs_sym(): eigval is an alias of eigvec" );
  
  const bool status = sp_auxlib::eigs_sym(eigval, eigvec, X, n_eigvals, form, tol);
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eigs_sym(): failed to converge", false);
    }
  
  return status;
  }



//! @}
