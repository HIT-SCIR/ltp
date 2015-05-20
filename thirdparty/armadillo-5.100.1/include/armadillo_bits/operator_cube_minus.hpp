// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_cube_minus
//! @{



//! unary -
template<typename T1>
arma_inline
const eOpCube<T1, eop_neg>
operator-
  (
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_neg>(X.get_ref());
  }



//! cancellation of two consecutive negations: -(-T1)
template<typename T1>
arma_inline
const typename ProxyCube<T1>::stored_type&
operator-
  (
  const eOpCube<T1, eop_neg>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return X.P.Q;
  }



//! BaseCube - scalar
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_minus_post>
operator-
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const typename T1::elem_type               k
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_minus_post>(X.get_ref(), k);
  }



//! scalar - BaseCube
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_minus_pre>
operator-
  (
  const typename T1::elem_type               k,
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_minus_pre>(X.get_ref(), k);
  }



//! complex scalar - non-complex BaseCube (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>
operator-
  (
  const std::complex<typename T1::pod_type>& k,
  const BaseCube<typename T1::pod_type, T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>('j', X.get_ref(), k);
  }



//! non-complex BaseCube - complex scalar (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>
operator-
  (
  const BaseCube<typename T1::pod_type, T1>& X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>('j', X.get_ref(), k);
  }



//! subtraction of BaseCube objects with same element type
template<typename T1, typename T2>
arma_inline
const eGlueCube<T1, T2, eglue_minus>
operator-
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const BaseCube<typename T1::elem_type,T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlueCube<T1, T2, eglue_minus>(X.get_ref(), Y.get_ref());
  }



//! subtraction of BaseCube objects with different element types
template<typename T1, typename T2>
inline
const mtGlueCube<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_minus>
operator-
  (
  const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
  const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  return mtGlueCube<out_eT, T1, T2, glue_mixed_minus>( X.get_ref(), Y.get_ref() );
  }



//! @}
