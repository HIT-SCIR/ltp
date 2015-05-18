// Copyright (C) 2010-2011 Conrad Sanderson
// Copyright (C) 2010-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_log_det
//! @{



//! log determinant of mat
template<typename T1>
inline
bool
log_det
  (
        typename T1::elem_type&          out_val,
        typename T1::pod_type&           out_sign,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return auxlib::log_det(out_val, out_sign, X);
  }



template<typename T1>
inline
void
log_det
  (
        typename T1::elem_type& out_val,
        typename T1::pod_type&  out_sign,
  const Op<T1,op_diagmat>&      X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const diagmat_proxy<T1> A(X.m);
  
  const uword N = A.n_elem;
  
  if(N == 0)
    {
    out_val  = eT(0);
    out_sign =  T(1);
    
    return;
    }
  
  eT x = A[0];
  
  T  sign = (is_complex<eT>::value == false) ? ( (access::tmp_real(x) < T(0)) ? -1 : +1 ) : +1;
  eT val  = (is_complex<eT>::value == false) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x ) : std::log(x);
  
  for(uword i=1; i<N; ++i)
    {
    x = A[i];
    
    sign *= (is_complex<eT>::value == false) ? ( (access::tmp_real(x) < T(0)) ? -1 : +1 ) : +1;
    val  += (is_complex<eT>::value == false) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x ) : std::log(x);
    }
  
  out_val  = val;
  out_sign = sign;
  }



//! @}
