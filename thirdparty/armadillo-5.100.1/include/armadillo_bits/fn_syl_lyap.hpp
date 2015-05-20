// Copyright (C) 2011-2012 Conrad Sanderson
// Copyright (C) 2011-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_syl_lyap
//! @{


//! find the solution of the Sylvester equation AX + XB = C
template<typename T1, typename T2, typename T3>
inline
bool
syl
  (
        Mat <typename T1::elem_type>   & out,
  const Base<typename T1::elem_type,T1>& in_A,
  const Base<typename T1::elem_type,T2>& in_B,
  const Base<typename T1::elem_type,T3>& in_C,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const unwrap_check<T1> tmp_A(in_A.get_ref(), out);
  const unwrap_check<T2> tmp_B(in_B.get_ref(), out);
  const unwrap_check<T3> tmp_C(in_C.get_ref(), out);
  
  const Mat<eT>& A = tmp_A.M;
  const Mat<eT>& B = tmp_B.M;
  const Mat<eT>& C = tmp_C.M;
  
  const bool status = auxlib::syl(out, A, B, C);
  
  if(status == false)
    {
    out.reset();
    arma_bad("syl(): equation appears to be singular", false);
    }
  
  return status;
  }



template<typename T1, typename T2, typename T3>
inline
Mat<typename T1::elem_type>
syl
  (
  const Base<typename T1::elem_type,T1>& in_A,
  const Base<typename T1::elem_type,T2>& in_B,
  const Base<typename T1::elem_type,T3>& in_C,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1> tmp_A( in_A.get_ref() );
  const unwrap<T2> tmp_B( in_B.get_ref() );
  const unwrap<T3> tmp_C( in_C.get_ref() );
  
  const Mat<eT>& A = tmp_A.M;
  const Mat<eT>& B = tmp_B.M;
  const Mat<eT>& C = tmp_C.M;
  
  Mat<eT> out;
  
  const bool status = auxlib::syl(out, A, B, C);
  
  if(status == false)
    {
    out.reset();
    arma_bad("syl(): equation appears to be singular");
    }
  
  return out;
  }



//! @}
