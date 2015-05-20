// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_cube_plus
//! @{



//! unary plus operation (does nothing, but is required for completeness)
template<typename T1>
arma_inline
const BaseCube<typename T1::elem_type,T1>&
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return X;
  }



//! BaseCube + scalar
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_plus>
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const typename T1::elem_type               k
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_plus>(X.get_ref(), k);
  }



//! scalar + BaseCube
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_plus>
operator+
  (
  const typename T1::elem_type               k,
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_plus>(X.get_ref(), k);
  }



//! non-complex BaseCube + complex scalar (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>
operator+
  (
  const BaseCube<typename T1::pod_type, T1>& X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>('j', X.get_ref(), k);
  }



//! complex scalar + non-complex BaseCube (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>
operator+
  (
  const std::complex<typename T1::pod_type>& k,
  const BaseCube<typename T1::pod_type, T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>('j', X.get_ref(), k);  // NOTE: order is swapped
  }



//! addition of BaseCube objects with same element type
template<typename T1, typename T2>
arma_inline
const eGlueCube<T1, T2, eglue_plus>
operator+
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const BaseCube<typename T1::elem_type,T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlueCube<T1, T2, eglue_plus>(X.get_ref(), Y.get_ref());
  }



//! addition of BaseCube objects with different element types
template<typename T1, typename T2>
inline
const mtGlueCube<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_plus>
operator+
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
  
  return mtGlueCube<out_eT, T1, T2, glue_mixed_plus>( X.get_ref(), Y.get_ref() );
  }



//! @}
