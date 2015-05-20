// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_det
//! @{



//! determinant of mat
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Base<typename T1::elem_type,T1>& X,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return auxlib::det(X, slow);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Base<typename T1::elem_type,T1>& X,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "det(): unknown method specified" );
  
  const bool slow = (sig == 's');
  
  return auxlib::det(X, slow);
  }



//! determinant of diagmat
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1, op_diagmat>& X,
  const bool slow = false
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(slow);
  
  typedef typename T1::elem_type eT;
  
  const diagmat_proxy<T1> A(X.m);
  
  const uword N = A.n_elem;
  
  eT val1 = eT(1);
  eT val2 = eT(1);
  
  uword i,j;
  for(i=0, j=1; j<N; i+=2, j+=2)
    {
    val1 *= A[i];
    val2 *= A[j];
    }
  
  
  if(i < N)
    {
    val1 *= A[i];
    }
  
  return val1 * val2;
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1, op_diagmat>& X,
  const char* method
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "det(): unknown method specified" );
  
  return det(X, false);
  }



//! determinant of a triangular matrix
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1, op_trimat>& X,
  const bool slow = false
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(slow);
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> P(X.m);
  
  const uword N = P.get_n_rows();
  
  arma_debug_check( (N != P.get_n_cols()), "det(): matrix is not square" );
  
  eT val1 = eT(1);
  eT val2 = eT(1);
  
  uword i,j;
  for(i=0, j=1; j<N; i+=2, j+=2)
    {
    val1 *= P.at(i,i);
    val2 *= P.at(j,j);
    }
  
  if(i < N)
    {
    val1 *= P.at(i,i);
    }
  
  return val1 * val2;
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1, op_trimat>& X,
  const char* method
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "det(): unknown method specified" );
  
  return det(X, false);
  }



//! determinant of inv(A), without doing the inverse operation
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1,op_inv>& X,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const eT tmp = det(X.m, slow);
  
  arma_debug_warn( (tmp == eT(0)), "det(): warning: denominator is zero" );
  
  return eT(1) / tmp;
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1,op_inv>& X,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "det(): unknown method specified" );
  
  const bool slow = (sig == 's');
  
  return det(X, slow);
  }



//! determinant of trans(A)
template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1,op_htrans>& in,
  const bool slow = false,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk1 = 0,
  const typename         arma_not_cx<typename T1::elem_type>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return auxlib::det(in.m, slow);  // bypass op_htrans
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
det
  (
  const Op<T1,op_htrans>& in,
  const char* method,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk1 = 0,
  const typename         arma_not_cx<typename T1::elem_type>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "det(): unknown method specified" );
  
  const bool slow = (sig == 's');
  
  return auxlib::det(in.m, slow);  // bypass op_htrans
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result &
det(const T& x)
  {
  return x;
  }



//! @}
