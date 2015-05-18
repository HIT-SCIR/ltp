// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Stanislav Funiak
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_eig_sym
//! @{


//! Eigenvalues of real/complex symmetric/hermitian matrix X
template<typename T1>
inline
bool
eig_sym
  (
         Col<typename T1::pod_type>&     eigval,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  // unwrap_check not used as T1::elem_type and T1::pod_type may not be the same.
  // furthermore, it doesn't matter if X is an alias of eigval, as auxlib::eig_sym() makes a copy of X
  
  const bool status = auxlib::eig_sym(eigval, X);
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eig_sym(): failed to converge", false);
    }
  
  return status;
  }



//! Eigenvalues of real/complex symmetric/hermitian matrix X
template<typename T1>
inline
Col<typename T1::pod_type>
eig_sym
  (
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  Col<typename T1::pod_type> out;
  const bool status = auxlib::eig_sym(out, X);

  if(status == false)
    {
    out.reset();
    arma_bad("eig_sym(): failed to converge");
    }
  
  return out;
  }



//! Eigenvalues and eigenvectors of real/complex symmetric/hermitian matrix X
template<typename T1> 
inline
bool
eig_sym
  (
         Col<typename T1::pod_type>&     eigval,
         Mat<typename T1::elem_type>&    eigvec,
  const Base<typename T1::elem_type,T1>& X,
  const char* method =                   "dc",
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'd')),         "eig_sym(): unknown method specified"     );
  arma_debug_check( void_ptr(&eigval) == void_ptr(&eigvec), "eig_sym(): eigval is an alias of eigvec" );
  
  const Proxy<T1> P(X.get_ref());
  
  const bool is_alias = P.is_alias(eigvec);
  
  Mat<eT>  eigvec_tmp;
  Mat<eT>& eigvec_out = (is_alias == false) ? eigvec : eigvec_tmp;
  
  bool status = false;
  
  if(sig == 'd')       { status = auxlib::eig_sym_dc(eigval, eigvec_out, P.Q); }
  
  if(status == false)  { status = auxlib::eig_sym(eigval, eigvec_out, P.Q);    }
  
  if(status == false)
    {
    eigval.reset();
    eigvec.reset();
    arma_bad("eig_sym(): failed to converge", false);
    }
  else
    {
    if(is_alias)  { eigvec.steal_mem(eigvec_tmp); }
    }
  
  return status;
  }



//! @}
